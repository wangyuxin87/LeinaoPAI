# py36-pytorch41-tf111-bashrc

## Packages
* cuda9.0
* cudnn7
* skimage
* PIL
* cv2
* matplotlib
* pytorch 0.4.1
* tensorflow 1.11.0

## Feature
- customizable [.bashrc](https://github.com/Chaway/LeinaoPAI/blob/master/images/py36-pytorch41-tf111-bashrc/.bashrc)  (Note this `.bashrc` is under `/userhome/root` , and its modification can be preserved)
- users can set own Environment Variables in `/userhome/root/.bashrc`. For example, 
```bash
 export HOME=/userhome/root
```
Then some customized and preservable configration files (e.g., `.vimrc`, `.vim`, `.bash_aliases`, `.inputrc`) will be supported under `/userhome/root` 

## Build
```bash
docker build -t py36-pytorch41-tf111-bashrc ./images/py36-pytorch41-tf111-bashrc/
```
