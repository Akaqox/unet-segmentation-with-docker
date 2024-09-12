
<div align='center'>
    <h1><b> Leaf Segmentation Project with Dockerized Anaconda Environment</b></h1>
<p>A segmentation model has been developed with ability to use multiple loss function options and customizable arguments. The model supports several configurations, including a flat U-Net, as well as U-Net variants with ResNet-34 and ResNet-50 encoders.</p>

<p>To ensure compatibility across different environments, the entire project has been containerized using Docker. This allows for a plug-and-play approach, simplifying the process of running the model in various setups.</p>
  
![Python](https://badgen.net/badge/Python/[3.11]/blue?) 
![Pytorch](https://badgen.net/badge/Pytorch/[2.4.0]/red?) 
</div>

---

## 💾 **ABOUT**

Will be added later

<br />

## Project Structure

Will be added later
  
## 💻 **TECHNOLOGIES**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)

##  **INSTALLATION**

```
git clone https://github.com/Akaqox/unet-segmentation-with-docker.git
docker build -t seg:latest .
docker run -v /opt/data/seg:/app/results --gpus all -it --ipc=host  seg

python -u main.py --bs --model unet50
python -u inference 
python -u inference --image 'path to image'
python -u inference --jv
```

## 🔎 **SHOWCASE**
 <h2><b> Will be added </b></h1>
<img src=/>
<br />
 <h2><b> </b></h1>

<br />

## 🔎 **REFERENCES**

<p></p>
<p></p>
<p></p>


<br />

---
