import argparse, torch, logging, os, shutil
from icefall.tts_datamodule import BakerZhTtsDataModule
from icefall.checkpoint import save_checkpoint
from icefall.utils import MetricsTracker
from lhotse.utils import fix_random_seed
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import utils
import utils.model
import typing

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def parse_args(args):
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="The seed for random generators intended for reproducibility",
	)
	parser.add_argument(
		"--dataset_dir",
		type=str,
		help="Dataset directory",
	)
	parser.add_argument(
		"--exp_dir",
		type=str,
		help="Directory to store all train logs and files.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=1000,
		help="Number of epochs to train",
	)
	parser.add_argument(
		"--pretrained_checkpoint",
		type=str,
		default=None,
		help="pretrained checkpoint to initialize from",
	)
	parser.add_argument(
		"--keep_epochs",
		type=int,
		default=10,
		help="Number of epochs to keep. -1 means all epochs.",
	)
	parser.add_argument(
		"--resume",
		type=bool,
		default=None,
		help="Resume training. Will ignore --pretrained_checkpoint"
	)
	parser.add_argument(
		"--save_every_n",
		type=int,
		default=10,
		help="Save checkpoint after processing this number of epochs periodically.",
	)
	parser.add_argument(
		"--use_fp16",
		type=bool,
		default=False,
		help="Whether to use half precision training.",
	)
	parser.add_argument(
		"--start_epoch",
		type=int,
		default=1,
		help="""Resume training from this epoch. It should be positive.
		If larger than 1, it will load checkpoint from
		exp-dir/epoch-{start_epoch-1}.pt
		""",
	)
	BakerZhTtsDataModule.add_arguments(parser)
	return parser.parse_args(args)

def GetModel(args):
	builder = utils.model.ModelBuilder()

	token_filename = os.path.join(args.dataset_dir, 'tokens.txt')
	builder.LoadTokenizer(token_filename)

	cmvn_filename = os.path.join(args.dataset_dir, 'cmvn.json')
	builder.LoadCMVN(cmvn_filename)

	return builder.BuildModel()

def prepare_input(batch: dict, tokenizer: Tokenizer, device: torch.device, params):
    """Parse batch data"""
    mel_mean = params.data_args.data_statistics.mel_mean
    mel_std_inv = 1 / params.data_args.data_statistics.mel_std
    for i in range(batch["features"].shape[0]):
        n = batch["features_lens"][i]
        batch["features"][i : i + 1, :n, :] = (
            batch["features"][i : i + 1, :n, :] - mel_mean
        ) * mel_std_inv
        batch["features"][i : i + 1, n:, :] = 0

    audio = batch["audio"].to(device)
    features = batch["features"].to(device)
    audio_lens = batch["audio_lens"].to(device)
    features_lens = batch["features_lens"].to(device)
    tokens = batch["tokens"]

    tokens = tokenizer.texts_to_token_ids(tokens, intersperse_blank=True)
    tokens = k2.RaggedTensor(tokens)
    row_splits = tokens.shape.row_splits(1)
    tokens_lens = row_splits[1:] - row_splits[:-1]
    tokens = tokens.to(device)
    tokens_lens = tokens_lens.to(device)
    # a tensor of shape (B, T)
    tokens = tokens.pad(mode="constant", padding_value=tokenizer.pad_id)

    max_feature_length = fix_len_compatibility(features.shape[1])
    if max_feature_length > features.shape[1]:
        pad = max_feature_length - features.shape[1]
        features = torch.nn.functional.pad(features, (0, 0, 0, pad))

        #  features_lens[features_lens.argmax()] += pad

    return audio, audio_lens, features, features_lens.long(), tokens, tokens_lens.long()


def train_one_epoch(
	args,
	model: utils.model.Model,
	optimizer: torch.optim.Optimizer,
	train_dl: torch.utils.data.DataLoader,
	valid_dl: torch.utils.data.DataLoader,
	scaler: GradScaler,
	tb_writer: typing.Union[SummaryWriter, None] = None,
	world_size: int = 1,
	rank: int = 0,
) -> None:
	"""Train the model for one epoch.

	The training loss from the mean of all frames is saved in
	`params.train_loss`. It runs the validation process every
	`params.valid_interval` batches.

	Args:
	  params:
		It is returned by :func:`get_params`.
	  model:
		The model for training.
	  optimizer:
		The optimizer.
	  train_dl:
		Dataloader for the training dataset.
	  valid_dl:
		Dataloader for the validation dataset.
	  scaler:
		The scaler used for mix precision training.
	  tb_writer:
		Writer to write log messages to tensorboard.
	"""
	model.train()
	device = model.device if isinstance(model, DDP) else next(model.parameters()).device
	get_losses = model.module.get_losses if isinstance(model, DDP) else model.get_losses

	# used to track the stats over iterations in one epoch
	tot_loss = MetricsTracker()

	saved_bad_model = False

	def save_bad_model(suffix: str = ""):
		save_checkpoint(
			filename=os.path.join(args.exp_dir, f"bad-model{suffix}-{rank}.pt"),
			model=model,
			params=model.GetParams(),
			optimizer=optimizer,
			scaler=scaler,
			rank=0,
		)

	for batch_idx, batch in enumerate(train_dl):
		model.GetParams().batch_idx_train += 1
		# audio: (N, T), float32
		# features: (N, T, C), float32
		# audio_lens, (N,), int32
		# features_lens, (N,), int32
		# tokens: List[List[str]], len(tokens) == N

		batch_size = len(batch["tokens"])

		(
			audio,
			audio_lens,
			features,
			features_lens,
			tokens,
			tokens_lens,
		) = prepare_input(batch, tokenizer, device, params)
		try:
			with autocast(enabled=params.use_fp16):
				losses = get_losses(
					{
						"x": tokens,
						"x_lengths": tokens_lens,
						"y": features.permute(0, 2, 1),
						"y_lengths": features_lens,
						"spks": None,  # should change it for multi-speakers
						"durations": None,
					}
				)

				loss = sum(losses.values())

				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

				loss_info = MetricsTracker()
				loss_info["samples"] = batch_size

				s = 0

				for key, value in losses.items():
					v = value.detach().item()
					loss_info[key] = v * batch_size
					s += v * batch_size

				loss_info["tot_loss"] = s

				tot_loss = tot_loss + loss_info
		except:  # noqa
			save_bad_model()
			raise

		if params.batch_idx_train % 100 == 0 and params.use_fp16:
			# If the grad scale was less than 1, try increasing it.
			# The _growth_interval of the grad scaler is configurable,
			# but we can't configure it to have different
			# behavior depending on the current grad scale.
			cur_grad_scale = scaler._scale.item()

			if cur_grad_scale < 8.0 or (
				cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0
			):
				scaler.update(cur_grad_scale * 2.0)
			if cur_grad_scale < 0.01:
				if not saved_bad_model:
					save_bad_model(suffix="-first-warning")
					saved_bad_model = True
				logging.warning(f"Grad scale is small: {cur_grad_scale}")
			if cur_grad_scale < 1.0e-05:
				save_bad_model()
				raise RuntimeError(
					f"grad_scale is too small, exiting: {cur_grad_scale}"
				)

		if params.batch_idx_train % params.log_interval == 0:
			cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

			logging.info(
				f"Epoch {params.cur_epoch}, batch {batch_idx}, "
				f"global_batch_idx: {params.batch_idx_train}, "
				f"batch size: {batch_size}, "
				f"loss[{loss_info}], tot_loss[{tot_loss}], "
				+ (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
			)

			if tb_writer is not None:
				loss_info.write_summary(
					tb_writer, "train/current_", params.batch_idx_train
				)
				tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
				if params.use_fp16:
					tb_writer.add_scalar(
						"train/grad_scale", cur_grad_scale, params.batch_idx_train
					)

		if params.batch_idx_train % params.valid_interval == 1:
			logging.info("Computing validation loss")
			valid_info = compute_validation_loss(
				params=params,
				model=model,
				tokenizer=tokenizer,
				valid_dl=valid_dl,
				world_size=world_size,
				rank=rank,
			)
			model.train()
			logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
			logging.info(
				"Maximum memory allocated so far is "
				f"{torch.cuda.max_memory_allocated()//1000000}MB"
			)
			if tb_writer is not None:
				valid_info.write_summary(
					tb_writer, "train/valid_", params.batch_idx_train
				)

	loss_value = tot_loss["tot_loss"] / tot_loss["samples"]
	params.train_loss = loss_value
	if params.train_loss < params.best_train_loss:
		params.best_train_epoch = params.cur_epoch
		params.best_train_loss = params.train_loss



def main(args):
	logging.info("Training started")
	rank = 1
	world_size = 1

	# Setup tensorboard logger
	tensorboard_logdir = os.path.join(args.exp_dir, "tensorboard")
	writer = SummaryWriter(log_dir=tensorboard_logdir)

	logging.info("About to create model")
	model = GetModel(args)

	num_param = sum([p.numel() for p in model.GetModel().parameters()])
	logging.info(f"Number of parameters: {num_param}")

	if args.pretrained_checkpoint is not None:
		checkpoint = model.LoadCheckpoint(args.pretrained_checkpoint)
	model.UploadToDevice(device)

	optimizer = torch.optim.Adam(model.parameters(), **model.optimizer)

	logging.info("About to create datamodule")
	baker_zh = BakerZhTtsDataModule(args)
	train_cuts = baker_zh.train_cuts()
	train_dl = baker_zh.train_dataloaders(train_cuts)
	valid_cuts = baker_zh.valid_cuts()
	valid_dl = baker_zh.valid_dataloaders(valid_cuts)

	scaler = GradScaler(enabled=args.use_fp16, init_scale=1.0)
	if args.pretrained_checkpoint is not None and "grad_scaler" in checkpoint:
		logging.info("Loading grad scaler state dict")
		scaler.load_state_dict(checkpoint["grad_scaler"])

	for epoch in range(args.start_epoch, args.epochs + 1):
		logging.info(f"Start epoch {epoch}")
		fix_random_seed(args.seed + epoch - 1)
		if getattr(train_dl, "sampler") is not None:
			train_dl.sampler.set_epoch(epoch - 1)

		model.GetParams().cur_epoch = epoch
		
		train_one_epoch(
			args=args,
			model=model,
			optimizer=optimizer,
			train_dl=train_dl,
			valid_dl=valid_dl,
			scaler=scaler,
			tb_writer=writer,
			world_size=world_size,
			rank=rank,
		)

		if epoch % args.save_every_n == 0 or epoch == args.epochs:
			filename = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
			save_checkpoint(
				filename=filename,
				params=model.Serialize(),
				model=model,
				optimizer=optimizer,
				scaler=scaler,
				rank=rank,
			)
			if args.keep_epochs != -1:
				model_list = [ x for x in os.listdir(args.output_dir) if x.startswith("epoch_") and x.endswith(".pt") ]
				model_list = sorted(model_list, key=lambda x: int(x[len('epoch_'):-3]))
				if len(model_list) > args.keep_epochs:
					for i in range(0, len(model_list) - args.keep_epochs):
						os.remove(os.path.join(args.output_dir, model_list[i]))

			if rank == 0:
				if model.GetParams().best_train_epoch == model.GetParams().cur_epoch:
					best_train_filename = os.path.join(args.dataset_dir, "best-train-loss.pt")
					shutil.copyfile(src=filename, dst=best_train_filename)

				if model.GetParams().best_valid_epoch == model.GetParams().cur_epoch:
					best_valid_filename = os.path.join(args.dataset_dir, "best-valid-loss.pt")
					shutil.copyfile(src=filename, dst=best_valid_filename)

	logging.info("Done!")

if __name__ == '__main__':
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)
	args = parse_args()
	fix_random_seed(args.seed)
	main(args)