# deep_dream_tensorflow
An implement of google deep dream with tensorflow. Code can be found [here](https://github.com/hjptriplebee/deep_dream_tensorflow).Detailed description can be found [here](http://blog.csdn.net/accepthjp/article/details/77882814)

Demo:

<p align="center">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/nature_image.jpg" width="30%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/nature_image_50.jpg" width="30%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/nature_image_500.jpg" width="30%">
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/paint.jpg" width="45%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/paint_50.jpg" width="45%">
</p>

# Requirement
- Python3
- OpenCV
- tensorflow 1.0

# Usage
-python3 main.py --input {input path} --output {output path}

If you don't input any image, it will generate a dream image with noise.

# Tips
Gradient ascent region has uncertainty, even same image with same parameters can generate different pictures.

<p align="center">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/nature_image.jpg" width="30%" name = "a">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/nature_image_500.jpg" width="30%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/nature_image_500_2.jpg" width="30%">
</p>

Larger "iter_num" means a more surprising and more different image.

<p align="center">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed5a_1x1_pre_relu_10.jpg" width="24%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed5a_1x1_pre_relu_50.jpg" width="24%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed5a_1x1_pre_relu_200.jpg" width="24%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed5a_1x1_pre_relu_1000.jpg" width="24%">
</p>

larger receptive field means more semantic information.

<p align="center">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed3b_pool_reduce_500.jpg" width="30%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed4c_pool_reduce_500.jpg" width="30%">
<img src="https://raw.githubusercontent.com/hjptriplebee/deep_dream_tensorflow/master/output/mixed5b_pool_reduce_500.jpg" width="30%">
</p>

To different image, best parameters are different.