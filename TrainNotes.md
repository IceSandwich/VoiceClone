# Setup environment / Infer

1. 克隆 icefall
2. 执行
``` bash
python egs\ljspeech\TTS\matcha\infer.py --epoch 2000 --exp-dir "D:\GITHUB\icefall-tts-baker-matcha-zh-2024-12-27" --tokens "D:\GITHUB\icefall-tts-baker-matcha-zh-2024-12-27\tokens.txt" --cmvn "D:\GITHUB\icefall-tts-baker-matcha-zh-2024-12-27\cmvn.json" --input_text "您好" --output-wav "./test-hell.wav" --vocoder "D:\GITHUB\icefall-tts-baker-matcha-zh-2024-12-27\generator_v1"
```

报错解决：
1. 软连接问题
``` txt
raceback (most recent call last):
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\infer.py", line 16, in <module>
    from tokenizer import Tokenizer
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\tokenizer.py", line 1
    ../vits/tokenizer.py
    ^
SyntaxError: invalid syntax
```
复制../vits/tokenizer.py替换matcha\tokenizer.py即可

2. 依赖问题
``` txt
Traceback (most recent call last):
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\infer.py", line 16, in <module>    
    from tokenizer import Tokenizer
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\tokenizer.py", line 20, in <module>
    import tacotron_cleaner.cleaners
ModuleNotFoundError: No module named 'tacotron_cleaner'
```
安装以下包
``` bash
python -m pip install espnet_tts_frontend
```

3. 依赖问题
``` txt
Traceback (most recent call last):
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\tokenizer.py", line 23, in <module>
    from piper_phonemize import phonemize_espeak
ModuleNotFoundError: No module named 'piper_phonemize'
```
这个包要自己编译，网上只有Linux版本。
克隆`piper_phonemize`：https://github.com/rhasspy/piper-phonemize
`piper_phonemize`依赖：https://github.com/rhasspy/espeak-ng 这个不用克隆。

接下来使用CMake安装`piper_phonemize`。其实我们不需要c++版本的piper，因为pip会自己再编译一次。我们使用CMake是因为依赖都在CMake里处理的，包括下载espeak-ng，这样更方便。

假设编译的文件夹是`D:\GITHUB\piper-phonemize\build`。

这样你就会发现文件夹`D:\GITHUB\piper-phonemize\build\ei`，这是espeak-ng安装出来的文件。

接下来更改`setup.py`：
``` python
import platform
from pathlib import Path
import shutil
import sys

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.install import install

# change path
_ESPEAK_NG_DATA_DIR = Path(R"D:\GITHUB\piper-phonemize\build\ei\share\espeak-ng-data")
_TASHKELL_MODEL = Path(R"D:\GITHUB\piper-phonemize\etc\libtashkeel_model.ort")
INSTALL_DIR = Path(R"D:\GITHUB\piper-phonemize\build\install")

__version__ = "1.2.0"

ext_modules = [
    Pybind11Extension(
        "piper_phonemize_cpp",
        [
            "src/python.cpp",
            "src/phonemize.cpp",
            "src/phoneme_ids.cpp",
            "src/tashkeel.cpp"
        ],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[str(INSTALL_DIR / "include")], # change this
        library_dirs=[str(INSTALL_DIR / "lib")], # change this
        libraries=["espeak-ng", "onnxruntime"],
    ),
]

class CustomInstallCommand(install): # add this
    """Customized setuptools install command to copy espeak-ng-data."""

    def run(self):
        # Call superclass install command
        super().run()

        # Check if the operating system is Windows
        if platform.system() != "Windows":
            print("Skipping custom installation steps: not running on Windows")
            return

        # Define the source directories for espeak-ng-data and libraries
        espeak_ng_data_dir = INSTALL_DIR /  "share" / "espeak-ng-data"
        espeak_ng_dll = INSTALL_DIR / "bin" / "espeak-ng.dll"
        onnxruntime_dll = INSTALL_DIR / "lib" / "onnxruntime.dll"

        # Define the target directories within the Python environment
        target_data_dir = (
            Path(sys.prefix) / "Lib" / "site-packages" / "piper_phonemize" / "espeak-ng-data"
        )
        target_lib_dir = Path(sys.prefix) / "Library" / "bin"

        # Copy espeak-ng-data directory
        shutil.copytree(espeak_ng_data_dir, target_data_dir, dirs_exist_ok=True)

        # Copy espeak-ng library from bin
        shutil.copy(espeak_ng_dll, target_lib_dir)

        # Copy ONNX Runtime library
        shutil.copy(onnxruntime_dll, target_lib_dir)

setup(
    name="piper_phonemize",
    version=__version__,
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/piper-phonemize",
    description="Phonemization libary used by Piper text to speech system",
    long_description="",
    packages=["piper_phonemize"],
	package_data={ # change this
		"piper_phonemize": [
			str(p) for p in _ESPEAK_NG_DATA_DIR.rglob("*")
		]
		+ [str(_TASHKELL_MODEL)]
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext, "install": CustomInstallCommand, }, # change this
    zip_safe=False,
    python_requires=">=3.7",
)
```

然后安装`piper_phonemize`
``` bash
python setup.py install
```

可能出现编码问题
``` txt
D:\GITHUB\piper-phonemize\src\phoneme_ids.hpp(1): warning C4819: 该文件包含不能在当前代码页(936)中表示的字符。请将该文件保存为 Unicode 格式以防止数据丢失
```
将`phoneme_ids.hpp`另存为UTF8 BOM文件。

上面的`CustomInstallCommand`是拷贝dll到虚拟环境的bin目录，如果发现dll导入问题，可以检查一下这个函数有没有成功执行。

4. 依赖问题
``` txt
Traceback (most recent call last):
  File "egs\ljspeech\TTS\matcha\infer.py", line 16, in <module>
    from tokenizer import Tokenizer
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\tokenizer.py", line 30, in <module>
    from utils import intersperse
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\utils.py", line 25, in <module>
    from lhotse.dataset.sampling.base import CutSampler
ModuleNotFoundError: No module named 'lhotse'
```
安装：
``` bash
python -m pip install git+https://github.com/lhotse-speech/lhotse
```

5. 依赖问题
``` txt
  File "egs\ljspeech\TTS\matcha\infer.py", line 17, in <module>
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\train.py", line 12, in <module>
    import k2
ModuleNotFoundError: No module named 'k2'
```
不要在[这里](https://k2-fsa.github.io/k2/installation/pre-compiled-cpu-wheels-windows/2.3.1.html)安装。
要CUDA版本的要自己编译。

6. 依赖问题
``` txt
Traceback (most recent call last):
  File "egs\ljspeech\TTS\matcha\infer.py", line 17, in <module>
    from train import get_model, get_params
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\train.py", line 18, in <module>
    from models.matcha_tts import MatchaTTS
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\models\matcha_tts.py", line 5, in <module>
    import monotonic_align as monotonic_align
  File "D:\GITHUB\icefall\egs\ljspeech\TTS\matcha\monotonic_align\__init__.py", line 4, in <module>
    from .core import maximum_path_c
ModuleNotFoundError: No module named 'monotonic_align.core'
```
进入`egs\ljspeech\TTS\matcha\monotonic_align`，执行
``` bash
python -m pip install -e .
```

# 数据集

1. 使用`generate_datasets.ipynb`，从3s音频生成数据集。

2. 使用`make_dataset.py`，将数据集转为训练用的数据集。

``` bash
python make_dataset.py  --wav_folder "D:\MyGithub\CloneVoice\tvboy_denoised_mandarin_seed34751218\dataset" --label_text "D:\MyGithub\CloneVoice\assets\mandarin.txt" --output_dir "D:\MyGithub\CloneVoice\tvboy_denoised_mandarin_seed34751218\inputs" --language "mandarin" --speaker "tvboy_denoised" --gender "boy" --split_validset 0.05 
```

# 训练

``` bash
python make_dataset.py --wav_folder "D:\MyGithub\VoiceClone\assets\dataset\tvboy_denoised_mandarin_seed34751218" --label_text "D:\MyGithub\VoiceClone\assets\text\mandarin.txt" --output_dir "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218\inputs" --language "mandarin" --speaker "tvboy_denoised" --gender "boy" --validset_ratio 0.05

python infer.py --model "D:\MyGithub\VoiceClone\assets\checkpoint\tvboy_denoised_mandarin_seed34751218\epoch-1000.pt" --dataset_dir "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218\inputs" --vocoder "D:\MyGithub\VoiceClone\assets\model\vocoder\generator_v2" --text "你好呀，有什么需要我帮忙的吗？" --output output.wav

python train.py --exp-dir "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218" --num-epochs 3 --num-workers 4 --manifest-dir "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218\inputs" --tokens "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218\inputs\tokens.txt" --cmvn "D:\MyGithub\VoiceClone\assets\train\tvboy_denoised_mandarin_seed34751218\inputs\cmvn.json" --num-buckets 20 --drop-last 0 --vocoder-checkpoint "D:\MyGithub\VoiceClone\assets\model\vocoder\generator_v2" --pretrained-checkpoint "D:\GITHUB\icefall-tts-baker-matcha-zh-2024-12-27\epoch-2000.pt"
```