import os
from pathlib import Path
import subprocess
import sys
import tarfile
from requests import get
import shutil

def build_detectron2(env):
    detectron2_path = os.path.join(Path.cwd().anchor, "detectron2")
                
    if not os.path.exists(detectron2_path):
        try:
            import git
            print("[Installer] Cloning detectron2 repository...")
            git.Repo.clone_from("https://github.com/stran9682/detectron2", detectron2_path)
        except Exception as e:
            print(f"[Installer] Error occurred while cloning detectron2 repository: {e}")

    
    # 3. Build & Install detectron2
    try: 
        print("[Installer] Building and installing Detectron2...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            detectron2_path,
            "--no-build-isolation",
        ], env=env)
        print("[Installer] Detectron2 installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[Installer] Error occurred while building detectron2: {e}")

    shutil.rmtree(detectron2_path)

def install_pose_2d(frankmocap_dir):
    import git

    # define the name of the directory to be created
    path = os.path.join(frankmocap_dir, "detectors")

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
        
    ### download human pose estimation git
    if not os.path.exists(path+'/body_pose_estimator'):
        if not os.path.exists(path+'/lightweight-human-pose-estimation.pytorch'):
            os.chdir(path)
            print('downloading lightweight-human-pose-estimation.pytorch')
            git.Git(path).clone('https://github.com/stran9682/lightweight-human-pose-estimation.pytorch.git')
            if not os.path.exists(path+'/body_pose_estimator'):
                os.rename('lightweight-human-pose-estimation.pytorch','body_pose_estimator')
        else:
            print('lightweight-human-pose-estimation.pytorch already exists')
            if not os.path.exists(path+'/body_pose_estimator'):
                os.chdir(path)
                os.rename('lightweight-human-pose-estimation.pytorch','body_pose_estimator')
                print('folder renamed to body_pose_estimator')


    path = os.path.join(frankmocap_dir, "extra_data", "body_module", "body_pose_estimator")
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        os.chdir(path)
    else:
        print ("Successfully created the directory %s " % path)
        os.chdir(path)


    ### falta baixar pre trained model

    url = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"
    filename = 'checkpoint_iter_370000.pth'

    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

def download_data_body_module(frankmocap_dir):

    path = os.path.join(frankmocap_dir, "extra_data", "body_module")

    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        os.chdir(path)
    else:
        print ("Successfully created the directory %s " % path)
        os.chdir(path)
        
    url = "http://visiondata.cis.upenn.edu/spin/data.tar.gz"
    filename = 'data.tar.gz'

    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

    tf = tarfile.open(filename)
    tf.extractall()

    folder_path = Path(os.path.join(path, "data_from_spin"))
    if not folder_path.is_dir():
        os.rename('data','data_from_spin')

    path = os.path.join(frankmocap_dir, "extra_data", "body_module", "pretrained_weights")
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        os.chdir(path)
    else:
        print ("Successfully created the directory %s " % path)
        os.chdir(path)
        
    print('####Downloading pretrained_weights')
    url = "https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt"
    filename = '2020_05_31-00_50_43-best-51.749683916568756.pt'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

    url = "https://dl.fbaipublicfiles.com/eft/fairmocap_data/body_module/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt"
    filename = 'smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

    path = os.path.join(frankmocap_dir, "extra_data", "body_module")
    os.chdir(path)

    print('####Downloading other data')
    url = "https://dl.fbaipublicfiles.com/eft/fairmocap_data/body_module/J_regressor_extra_smplx.npy"
    filename = 'J_regressor_extra_smplx.npy'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)


def install_hand_detectors(frankmocap_dir, env):
    import git

    path = os.path.join(frankmocap_dir, "detectors")

    if not os.path.exists(os.path.join(path, "hand_object_detector")):
        folder_hand_object = os.path.join(Path.cwd().anchor, "hand_object_detector")

        try:
            print('[Installer] cloning hand object detector')
            git.Repo.clone_from('https://github.com/stran9682/hand_object_detector.git', folder_hand_object)
        except Exception as e:
            print(f"[Installer] Error occurred while cloning hand object detector repository: {e}")

        lib_path = os.path.join(folder_hand_object, "lib")
        try: 
            print("[Installer] Building hand object detector...")
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                lib_path,
                "--no-build-isolation",
            ], env=env)
            print("[Installer] Hand object detector installation completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"[Installer] Error occurred while building Hand object detector: {e}")

        shutil.move(folder_hand_object, os.path.join(path, "hand_object_detector"))    


    folder_orig = 'hand_detector.d2'
    folder_dest= 'hand_only_detector'
    ### Install 100-DOH hand-only detectors
    if not os.path.exists(path+'/'+folder_dest):
        if not os.path.exists(path+'/'+folder_orig):
            os.chdir(path)
            print('downloading '+folder_orig)
            git.Git(path).clone('https://github.com/ddshan/hand_detector.d2.git')
            if not os.path.exists(path+'/'+folder_dest):
                os.rename(folder_orig,folder_dest)
        else:
            print(folder_orig+' already exists')
            if not os.path.exists(path+'/'+folder_dest):
                os.chdir(path)
                os.rename(folder_orig,folder_dest)
                print('folder renamed to '+folder_dest)

    print("[Installer] installing hand detector weights from huggingface")      
    path = os.path.join(frankmocap_dir, "extra_data", "hand_module", "hand_detector")
    if not os.path.exists(path):
        hand_dectector_weights_path = os.path.join(Path.cwd().anchor, "hand_detector")

        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="ragamounibatchu/frankmocap-hand-detector-weights", local_dir=hand_dectector_weights_path)

        shutil.move(hand_dectector_weights_path, path)

        print("[Installer] weights installed sucessfully")
    else:
        print("[Installer] weights installed already")

def download_data_hand_module (frankmocap_dir):
    path = os.path.join(frankmocap_dir, "extra_data", "hand_module")

    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        os.chdir(path)
    else:
        print ("Successfully created the directory %s " % path)
        os.chdir(path)

    #### downloading other data
    url = "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/SMPLX_HAND_INFO.pkl"
    filename = 'SMPLX_HAND_INFO.pkl'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

    url = "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/mean_mano_params.pkl"
    filename = 'mean_mano_params.pkl'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

    path = os.path.join(frankmocap_dir, "extra_data", "hand_module", "pretrained_weights")
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        os.chdir(path)
    else:
        print ("Successfully created the directory %s " % path)
        os.chdir(path)

    url = "https://dl.fbaipublicfiles.com/eft/fairmocap_data/hand_module/checkpoints_best/pose_shape_best.pth"
    filename = 'pose_shape_best.pth'
    if not os.path.exists(path+'/'+filename):
        print ('Downloading: '+filename)
        download(url,filename)

def install_models(frankmocap_dir):
    print("[Installer] installing smpl models from huggingface")      
    path = os.path.join(frankmocap_dir, "extra_data", "smpl")
    if not os.path.exists(path):
        model_path = os.path.join(Path.cwd().anchor, "smpl")

        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="stran9682/smplx_model", local_dir=model_path)

        shutil.move(model_path, path)

        print("[Installer] models installed sucessfully")
    else:
        print("[Installer] models installed already")

def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)