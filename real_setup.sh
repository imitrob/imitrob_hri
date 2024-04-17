#!/bin/bash
cd ~
git clone https://github.com/imitrob/crow-base.git
cd ~/crow-base/src

git clone https://github.com/imitrob/teleop_gesture_toolbox.git --branch MM24
git clone https://github.com/imitrob/coppelia_sim_ros_interface.git --branch MM24
git clone https://github.com/petrvancjr/coppelia_sim_ros_interface_msgs.git
git clone https://github.com/imitrob/context_based_gesture_operation.git --branch MM24
git clone https://github.com/splintered-reality/py_trees_ros.git
git clone https://github.com/splintered-reality/py_trees_ros_interfaces.git
git clone https://github.com/imitrob/imitrob_templates.git
git clone https://github.com/imitrob/imitrob_hri.git
git clone https://github.com/IntelRealSense/realsense-ros.git
# shortcut using vcstool
#pip install vcstool
#vcs import < imitrob_hri/dependencies.repos

export ws=/home/$USER/crow-base
export condaenv="crow_env"

# Coppelia Sim install
mkdir -p $ws/src
cd $ws/src
wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 CoppeliaSim
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
cd CoppeliaSim
rm libsimExtROS2Interface.so # Official ROS2 interface needs to be disabled

# Leap Motion Controller install
sudo apt update
cd ~
gdown https://drive.google.com/uc?id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD
tar -xvzf Leap_Motion_SDK_Linux_2.3.1.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/
cd ..
rm -r LeapDeveloperKit_2.3.1+31549_linux
rm Leap_Motion_SDK_Linux_2.3.1.tgz

# Packages
conda install mamba -c conda-forge # Install mamba

mamba create -n $condaenv python=3.11
mamba activate $condaenv

cd $ws/
mamba env update -n $condaenv --file environment.yaml

# ROS2 env build
#cd $ws
#rosdep init
#rosdep update
#rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
#source $ws/install/setup.bash


cd ~
git clone https://github.com/imitrob/PyRep.git
cd ~/PyRep
alias cop='export COPPELIASIM_ROOT=$HOME/CoppeliaSim; export QT_QPA_PLATFORM_PLUGIN_PATH=$HOME/CoppeliaSim'
pip install .

cd ~
git clone https://github.com/petrvancjr/LeapAPI.git
cd ~/LeapAPI/Leap3.11
pip install .
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/LeapAPI/Leap3.11/
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/LeapAPI/Leap3.11/" >> ~/.bashrc


cd $ws/src/teleop_gesture_toolbox/include/data/trained_networks
gdown https://drive.google.com/uc?id=1cDtVfOL20f3YVJ6n_d-AQPwjVNe_nlf7

cd $ws/src/knowl
pip install .



alias crow="conda deactivate; conda activate $condaenv; source $ws/install/setup.bash; export ROS_DOMAIN_ID=77; export CROW_CONFIG_DIR=$ws/config; export CROW_LIB_DIR=$ws/lib; export CROW_SETUP_FILE=setup_404.yaml;"
alias crowrun="crow; cd $ws/user_scripts; python tmux_all.py --config run_tmux_404.yaml; tmux attach-session -t crow_run_all"




