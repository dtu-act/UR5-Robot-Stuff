<<<<<<< HEAD
### UR5 Robot Stuff
=======
###UR5 Robot Stuff
>>>>>>> main

Please also keep in mind [this](https://academy.universal-robots.com/media/jiehhszc/ursim_vmoracle_installation_guidev03_en.pdf) guide


1.  go to the "Universal Robots" website at [this](https://www.universal-robots.com/download/software-cb-series/simulator-non-linux/offline-simulator-cb-series-non-linux-ursim-3143/) link, download the URsim zip file and follow the instructions for [VirtualBox](https://www.virtualbox.org/wiki/Downloads)

2. (Optional) Create shared folder from as in [this](https://medium.com/macoclock/share-folder-between-macos-and-ubuntu-4ce84fb5c1ad) guide if you want to use files from your pc directly on the VM

4. Establish a network connection [guide](https://alainber.medium.com/virtualbox-networking-setup-1954c40e41f3) between your computer and the VM in order to control the robot arm simulation on the VM from your PC

5. Use python script 'move_ur5_in_grid.py' and consult its documentation to move the robot in a predefined grid assuming it is restricted to move within the range defined by the spherical grid defined in the aforementioned script. There may be issues if a grid is defined \textit{outside} of this sphere.