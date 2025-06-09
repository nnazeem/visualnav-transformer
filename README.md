# Setting up General Navigation Models over GCP, via RDP

Workflow for setting up a Google Cloud Platform (GCP) instance to run the `visualnav-transformer` project with a simulated TurtleBot 2 in Gazebo.

## Part 1: Environment Setup

This covers the creation and configuration of the entire software stack from scratch.

### 1.1: GCP Instance Setup

1.  **Create a VM Instance:**

      * In the GCP Console, go to **Compute Engine \> VM instances** and click **Create Instance**.
      * **Machine Configuration:**
          * Select a region with NVIDIA L4 GPUs.
          * Add 1 x **NVIDIA L4** GPU.
          * Choose an **N2** or **E2** series machine with at least 4 (chose 8) vCPUs and 16 GB (chose 32 GB) of RAM.
      * **Boot Disk:**
          * Click **Change**. Select **Ubuntu** as the OS and **Ubuntu 20.04 LTS** as the version.
          * Increase the disk size to at least **50 GB**.
      * **Firewall:** Check the boxes to **Allow HTTP traffic** and **Allow HTTPS traffic**.
      * Create the instance.

2.  **Install Core Dependencies:**

      * Connect to your instance using the **SSH-in-browser** window from the GCP console.
      * Update the system:
        ```bash
        sudo apt-get update && sudo apt-get upgrade -y
        ```
      * Install ROS Noetic (Desktop-Full version is recommended to get all tools):
        ```bash
        sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
        sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
        sudo apt-get update
        sudo apt-get install -y ros-noetic-desktop-full
        echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
        ```
      * Install additional required libraries we discovered:
        ```bash
        sudo apt-get install -y python3-pip python3-rosdep libfmt-dev ros-noetic-joy
        ```

3.  **Set up Conda Environment:**

      * Download and install Miniconda:
        ```bash
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh 
        # (Accept all defaults, say 'yes' to running conda init)
        ```
      * Close and reopen your SSH-in-browser window for the changes to take effect.
      * Clone the project repository:
        ```bash
        git clone https://github.com/robodhruv/visualnav-transformer.git
        ```
      * Create and activate the Conda environment from the project's file:
        ```bash
        cd visualnav-transformer
        conda env create -f deployment/deployment_environment.yaml
        conda activate vint_deployment
        ```
      * **Install all missing/conflicting Python packages with specific versions:**
        ```bash
        pip install pyyaml matplotlib einops vit-pytorch wandb prettytable opencv-python defusedxml "huggingface_hub==0.11.1" "empy==3.3.4"
        ```

4.  **Build the TurtleBot Workspace from Source:**

      * Create a Catkin workspace:
        ```bash
        mkdir -p ~/turtlebot_ws/src
        cd ~/turtlebot_ws/src
        ```
      * Clone all necessary source code (including dependencies for the dependencies):
        ```bash
        git clone https://github.com/hanruihua/Turtlebot2_on_Noetic.git
        cd Turtlebot2_on_Noetic
        sh turtlebot_noetic.sh
        cd .. 
        git clone https://github.com/strasdat/Sophus.git
        ```
      * Install the official `cmake` binary (the `apt` version is too old):
        ```bash
        cd ~
        wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.sh
        chmod +x cmake-3.29.3-linux-x86_64.sh
        sudo ./cmake-3.29.3-linux-x86_64.sh --prefix=/usr/local --exclude-subdir
        # (Accept the license when prompted)
        ```
      * Manually build and install the `Sophus` library:
        ```bash
        cd ~/turtlebot_ws/src/Sophus
        git checkout 1.22.10 # Use a version compatible with system libraries
        mkdir build && cd build
        cmake ..
        make -j$(nproc)
        sudo make install
        ```
      * Build the entire ROS workspace (this will take several minutes):
        ```bash
        cd ~/turtlebot_ws
        rosdep install --from-paths src --ignore-src -r -y
        catkin_make_isolated
        ```
      * Add the new workspace to your environment permanently:
        ```bash
        echo "source ~/turtlebot_ws/devel_isolated/setup.bash" >> ~/.bashrc
        ```

5.  **Fix the Project Scripts:**

      * The project scripts hardcode the wrong camera topic for Gazebo (assume USB connection to Kobuki Robot Platform). Edit the central `topic_names.py` file to fix this for all other scripts.
        ```bash
        cd ~/visualnav-transformer/deployment/src/
        nano topic_names.py
        ```
      * Find the line `IMAGE_TOPIC = "/usb_cam/image_raw"` and change it to use (can pu told one in bash comment instead of replacing entirely):
        `IMAGE_TOPIC = "/camera/rgb/image_raw"`
      * Save and exit the editor.

### 1.2: Remote Desktop (RDP) Setup

To view the GUI, RDP seemed straightforward over `ssh -X` or `VNC` .

1.  **Install Desktop & RDP Server (in SSH-in-browser):**
    ```bash
    sudo apt-get install -y xfce4 xfce4-goodies
    sudo apt-get install -y xrdp
    sudo adduser xrdp ssl-cert
    ```
2.  **Configure the Session:** Create a file to tell RDP to launch the XFCE desktop.
    ```bash
    echo xfce4-session > ~/.xsession
    sudo systemctl restart xrdp
    ```
3.  **Set a Password:** RDP uses your Linux user password. You must set one.
    ```bash
    sudo passwd <username> 
    # (Enter your new password when prompted)
    ```
4.  **GCP Firewall Rule:** In the GCP Console under **VPC Network \> Firewall**, create a new rule:
      * **Name:** `allow-rdp`
      * **Direction:** `Ingress`
      * **Action:** `Allow`
      * **Targets:** `All instances in the network`
      * **Source IPv4 ranges:** `0.0.0.0/0`
      * **Protocols and ports:** Check **TCP** and enter port `3389`.

## Part 2: Running the Navigation Demo

This is the final end-to-end workflow, performed inside the RDP session.

1.  **Connect to Your Instance:** Use the "Remote Desktop Connection" app on your local Windows computer to connect to your instance's External IP. Log in with your username and the password you just set.

2.  **Open Terminals:** Inside the remote desktop, open three terminal windows.

3.  **The "Golden Rule":** In **all three terminals**, run this command to set up the environment (or  ` >> .bashrc` so terminal will automaticall open with this done and run `exec bash` for it to take effect):

    ```bash
    conda activate vint_deployment && source ~/turtlebot_ws/devel_isolated/setup.bash
    ```

4.  **Record a Path:**

      * **Terminal 1:** Launch the simulator: `roslaunch turtlebot_gazebo turtlebot_world.launch`
      * **Terminal 2:** Start the recorder: `rosbag record -O ~/my_demo_path.bag /camera/rgb/image_raw /odom`
      * **Terminal 3:** Start the driver: `roslaunch turtlebot_teleop keyboard_teleop.launch`
      * Drive the robot in Gazebo, making sure to stop far from the origin. When done, press `Ctrl+C` in Terminal 2.

5.  **Create the Map:**

      * In Terminal 2 (or a new one with the environment set up), run:
        ```bash
        cd ~/visualnav-transformer/deployment/src/
        python create_topomap.py --dt 1 --dir my_demo_map ~/my_demo_path.bag
        ```

6.  **Run Autonomous Navigation:**

      * First, count the number of images to determine your goal node:
        ```bash
        ls ~/visualnav-transformer/deployment/topomaps/images/my_demo_map/ | wc -l
        ```
      * Subtract 1 from that number to get your `<goal_node_number>`.
      * Make sure the simulator is still running in Terminal 1 (restart it if needed to reset the robot's position).
      * In Terminal 2, launch the AI:
        ```bash
        cd ~/visualnav-transformer/deployment/src/
        ./navigate.sh "--model <gnm|vint|nomad> --dir my_demo_map --goal-node <goal_node_number>"
        ```
      * Watch the Gazebo window to see the robot navigate autonomously.

-----

# General Navigation Models: GNM, ViNT and NoMaD

**Contributors**: Dhruv Shah, Ajay Sridhar, Nitish Dashora, Catherine Glossop, Kyle Stachowicz, Arjun Bhorkar, Kevin Black, Noriaki Hirose, Sergey Levine

_Berkeley AI Research_

[Project Page](https://general-navigation-models.github.io) | [Citing](https://github.com/robodhruv/visualnav-transformer#citing) | [Pre-Trained Models](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

General Navigation Models are general-purpose goal-conditioned visual navigation policies trained on diverse, cross-embodiment training data, and can control many different robots in zero-shot. They can also be efficiently fine-tuned, or adapted, to new robots and downstream tasks. Our family of models is described in the following research papers (and growing):
1. [GNM: A General Navigation Model to Drive Any Robot](https://sites.google.com/view/drive-any-robot) (_October 2022_, presented at ICRA 2023)
2. [ViNT: A Foundation Model for Visual Navigation](https://general-navigation-models.github.io/vint/index.html) (_June 2023_, presented at CoRL 2023)
3. [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html) (_October 2023_)

## Overview
This repository contains code for training our family of models with your own data, pre-trained model checkpoints, as well as example code to deploy it on a TurtleBot2/LoCoBot robot. The repository follows the organization from [GNM](https://github.com/PrieureDeSion/drive-any-robot).

- `./train/train.py`: training script to train or fine-tune the ViNT model on your custom data.
- `./train/vint_train/models/`: contains model files for GNM, ViNT, and some baselines.
- `./train/process_*.py`: scripts to process rosbags or other formats of robot trajectories into training data.
- `./deployment/src/record_bag.sh`: script to collect a demo trajectory as a ROS bag in the target environment on the robot. This trajectory is subsampled to generate a topological graph of the environment.
- `./deployment/src/create_topomap.sh`: script to convert a ROS bag of a demo trajectory into a topological graph that the robot can use to navigate.
- `./deployment/src/navigate.sh`: script that deploys a trained GNM/ViNT/NoMaD model on the robot to navigate to a desired goal in the generated topological graph. Please see relevant sections below for configuration settings.
- `./deployment/src/explore.sh`: script that deploys a trained NoMaD model on the robot to randomly explore its environment. Please see relevant sections below for configuration settings.

## Train

This subfolder contains code for processing datasets and training models from your own data.

### Pre-requisites

The codebase assumes access to a workstation running Ubuntu (tested on 18.04 and 20.04), Python 3.7+, and a GPU with CUDA 10+. It also assumes access to conda, but you can modify it to work with other virtual environment packages, or a native setup.
### Setup
Run the commands below inside the `vint_release/` (topmost) directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate vint_train
    ```
3. Install the vint_train packages:
    ```bash
    pip install -e train/
    ```
4. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```


### Data-Wrangling
In the [papers](https://general-navigation-models.github.io), we train on a combination of publicly available and unreleased datasets. Below is a list of publicly available datasets used for training; please contact the respective authors for access to the unreleased data.
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2 (Modified)](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

We recommend you to download these (and any other datasets you may want to train on) and run the processing steps below.

#### Data Processing 

We provide some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
2. Run `data_split.py` on your dataset folder with the relevant args.

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:
- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `vint_release/train/vint_train/data/data_splits/`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### Training your General Navigation Models
Run this inside the `vint_release/train` directory:
```bash
python train.py -c <path_of_train_config_file>
```
The premade config yaml files are in the `train/config` directory. 

#### Custom Config Files
You can use one of the premade yaml files as a starting point and change the values as you need. `config/vint.yaml` is good choice since it has commented arguments. `config/defaults.yaml` contains the default config values (don't directly train with this config file since it does not specify any datasets for training).

#### Custom Datasets
Make sure your dataset and data-split directory follows the structures provided in the [Data Processing](#data-processing) section. Locate `train/vint_train/data/data_config.yaml` and append the following:

```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

Locate your training config file and add the following text under the `datasets` argument (feel free to change the values of `end_slack`, `goals_per_obs`, and `negative_mining`):
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # how many timesteps to cut off from the end of each trajectory  (in case many trajectories end in collisions)
    goals_per_obs: 1 # how many goals are sampled per observation
    negative_mining: True # negative mining from the ViNG paper (Shah et al.)
```

#### Training your model from a checkpoint
Instead of training from scratch, you can also load an existing checkpoint from the published results.
Add `load_run: <project_name>/<log_run_name>`to your .yaml config file in `vint_release/train/config/`. The `*.pth` of the file you are loading to be saved in this file structure and renamed to “latest”: `vint_release/train/logs/<project_name>/<log_run_name>/latest.pth`. This makes it easy to train from the checkpoint of a previous run since logs are saved this way by default. Note: if you are loading a checkpoint from a previous run, check for the name the run in the `vint_release/train/logs/<project_name>/`, since the code appends a string of the date to each run_name specified in the config yaml file of the run to avoid duplicate run names. 


If you want to use our checkpoints, you can download the `*.pth` files from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


## Deployment
This subfolder contains code to load a pre-trained ViNT and deploy it on the open-source [LoCoBot indoor robot platform](http://www.locobot.org/) with a [NVIDIA Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=asc_df_B0BZJTQ5YP/?tag=hyprod-20&linkCode=df0&hvadid=652427572954&hvpos=&hvnetw=g&hvrand=12520404772764575478&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1013585&hvtargid=pla-2112361227514&psc=1&gclid=CjwKCAjw4P6oBhBsEiwAKYVkq7dqJEwEPz0K-H33oN7MzjO0hnGcAJDkx2RdT43XZHdSWLWHKDrODhoCmnoQAvD_BwE). It can be easily adapted to be run on alternate robots, and researchers have been able to independently deploy it on the following robots – Clearpath Jackal, DJI Tello, Unitree A1, TurtleBot2, Vizbot – and in simulated environments like CARLA.

### LoCoBot Setup

This software was tested on a LoCoBot running Ubuntu 20.04.


#### Software Installation (in this order)
1. ROS: [ros-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
2. ROS packages: 
    ```bash
    sudo apt-get install ros-noetic-usb-cam ros-noetic-joy
    ```
3. [kobuki](http://wiki.ros.org/kobuki/Tutorials/Installation)
4. Conda 
    - Install anaconda/miniconda/etc. for managing environments
    - Make conda env with environment.yml (run this inside the `vint_release/` directory)
        ```bash
        conda env create -f deployment/deployment_environment.yaml
        ```
    - Source env 
        ```bash
        conda activate vint_deployment
        ```
    - (Recommended) add to `~/.bashrc`: 
        ```bash
        echo “conda activate vint_deployment” >> ~/.bashrc 
        ```
5. Install the `vint_train` packages (run this inside the `vint_release/` directory):
    ```bash
    pip install -e train/
    ```
6. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
7. (Recommended) Install [tmux](https://github.com/tmux/tmux/wiki/Installing) if not present.
    Many of the bash scripts rely on tmux to launch multiple screens with different commands. This will be useful for debugging because you can see the output of each screen.

#### Hardware Requirements
- LoCoBot: http://locobot.org (just the navigation stack)
- A wide-angle RGB camera: [Example](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W). The `vint_locobot.launch` file uses camera parameters that work with cameras like the ELP fisheye wide angle, feel free to modify to your own. Adjust the camera parameters in `vint_release/deployment/config/camera.yaml` your camera accordingly (used for visualization).
- [Joystick](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[keyboard teleop](http://wiki.ros.org/teleop_twist_keyboard) that works with Linux. Add the index mapping for the _deadman_switch_ on the joystick to the `vint_release/deployment/config/joystick.yaml`. You can find the mapping from buttons to indices for common joysticks in the [wiki](https://wiki.ros.org/joy). 


### Loading the model weights

Save the model weights *.pth file in `vint_release/deployment/model_weights` folder. Our model's weights are in [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).

### Collecting a Topological Map

_Make sure to run these scripts inside the `vint_release/deployment/src/` directory._


This section discusses a simple way to create a topological map of the target environment for deployment. For simplicity, we will use the robot in “path-following” mode, i.e. given a single trajectory in an environment, the task is to follow the same trajectory to the goal. The environment may have new/dynamic obstacles, lighting variations etc.

#### Record the rosbag: 
```bash
./record_bag.sh <bag_name>
```

Run this command to teleoperate the robot with the joystick and camera. This command opens up three windows 
1. `roslaunch vint_locobot.launch`: This launch file opens the `usb_cam` node for the camera, the joy node for the joystick, and nodes for the robot’s mobile base.
2. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
3. `rosbag record /usb_cam/image_raw -o <bag_name>`: This command isn’t run immediately (you have to click Enter). It will be run in the vint_release/deployment/topomaps/bags directory, where we recommend you store your rosbags.

Once you are ready to record the bag, run the `rosbag record` script and teleoperate the robot on the map you want the robot to follow. When you are finished with recording the path, kill the `rosbag record` command, and then kill the tmux session.

#### Make the topological map: 
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

This command opens up 3 windows:
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/vint_release/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 1.5 <bag_filename>`: This command plays the rosbag at x5 speed, so the python script is actually recording nodes 1.5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension. You can change this value in the `make_topomap.sh` file. The command does not run until you hit Enter, which you should only do once the python script gives its waiting message. Once you play the bag, move to the screen where the python script is running so you can kill it when the rosbag stops playing.

When the bag stops playing, kill the tmux session.


### Running the model 
#### Navigation
_Make sure to run this script inside the `vint_release/deployment/src/` directory._

```bash
./navigate.sh “--model <model_name> --dir <topomap_dir>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `vint_release/deployment/config/models.yaml` file. In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `vint_release/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `vint_release/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_locobot.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base).
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic.
3. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session. If you want to take control of the robot while it is navigating, the `joy_teleop.py` script allows you to do so with the joystick.

#### Exploration
_Make sure to run this script inside the `vint_release/deployment/src/` directory._

```bash
./exploration.sh “--model <model_name>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing).


The `<model_name>` is the name of the model in the `vint_release/deployment/config/models.yaml` file (note that only NoMaD works for exploration). In this file, you specify these parameters of the model for each model (defaults used):
- `config_path` (str): path of the *.yaml file in `vint_release/train/config/` used to train the model
- `ckpt_path` (str): path of the *.pth file in `vint_release/deployment/model_weights/`


Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `vint_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch vint_locobot.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base.
2. `python explore.py --model <model_name>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes exploration actions to the `/waypoint` topic.
3. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session. If you want to take control of the robot while it is navigating, the `joy_teleop.py` script allows you to do so with the joystick.


### Adapting this code to different robots

We hope that this codebase is general enough to allow you to deploy it to your favorite ROS-based robots. You can change the robot configuration parameters in `vint_release/deployment/config/robot.yaml`, like the max angular and linear velocities of the robot and the topics to publish to teleop and control the robot. Please feel free to create a Github Issue or reach out to the authors at shah@cs.berkeley.edu.


## Citing
```
@inproceedings{shah2022gnm,
  author    = {Dhruv Shah and Ajay Sridhar and Arjun Bhorkar and Noriaki Hirose and Sergey Levine},
  title     = {{GNM: A General Navigation Model to Drive Any Robot}},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.03370}
}

@inproceedings{shah2023vint,
  title     = {Vi{NT}: A Foundation Model for Visual Navigation},
  author    = {Dhruv Shah and Ajay Sridhar and Nitish Dashora and Kyle Stachowicz and Kevin Black and Noriaki Hirose and Sergey Levine},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2306.14846}
}

@article{sridhar2023nomad,
  author  = {Ajay Sridhar and Dhruv Shah and Catherine Glossop and Sergey Levine},
  title   = {{NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration}},
  journal = {arXiv pre-print},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.xxxx}
}
```
