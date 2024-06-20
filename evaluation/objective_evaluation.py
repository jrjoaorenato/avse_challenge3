'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

# import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
import csv
import json
from soundfile import SoundFile
from pesq import pesq
from pystoi import stoi
from concurrent.futures import ProcessPoolExecutor

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_pesq(target, enhanced, sr, mode):
    """Compute PESQ from: https://github.com/ludlows/python-pesq/blob/master/README.md
        Args:
            target (string): Name of file to read
            enhanced (string): Name of file to read
            sr (int): sample rate of files
            mode (string): 'wb' = wide-band (16KHz); 'nb' narrow-band (8KHz)
        Returns:
            PESQ metric (float)
                """
    return pesq(sr, target, enhanced, mode)

def run_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
           Args:
               target (string): Name of file to read
               enhanced (string): Name of file to read
               sr (int): sample rate of files
           Returns:
               STOI metric (float)
                   """
    return stoi(target, enhanced, sr)

def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
            Args:
                filename (string): Name of file to read
            Returns:
                ndarray: audio signal
            """
    try:
        wave_file = SoundFile(filename)
    except:
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.")
    return wave_file.read()

def run_metrics(scene_name, enhanced, target):

    # Retrieve the scene name

    enh_file = os.path.join(enhanced, f"{scene_name}.wav")
    tgt_file = os.path.join(target, f"{scene_name}_target.wav")
    scene_metrics_file = os.path.join('/home/jesus10/jrenato/avse/objective_evaluation/', f"{scene_name}.csv")

    # Skip processing with files dont exist or metrics have already been computed
    if ( not os.path.isfile(enh_file) ) or ( not os.path.isfile(tgt_file) ) or ( os.path.isfile(scene_metrics_file)) :
        return

    # Read enhanced signal
    enh = read_audio(enh_file)
    # Read clean/target signal
    clean = read_audio(tgt_file)

    # Check that both files are the same length, otherwise computing the metrics results in an error
    if len(clean) != len(enh):
        raise Exception(
            f"Wav files {enh_file} and {tgt_file} should have the same length"
        )

    # Compute metrics
    m_stoi = run_stoi(clean, enh, 16000)
    m_pesq = run_pesq(clean, enh, 16000, 'wb')

    # Store scene metrics in a tmp file
    with open(scene_metrics_file, "w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([scene_name, m_stoi, m_pesq])

# @hydra.main(config_path=".", config_name="config")
def compute_metrics():
    # paths to data
    enhanced = os.path.join('/home/jesus10/jrenato/avse/output/dev/')
    target = os.path.join('/home/jesus10/jrenato/avse/data/dev/scenes/')
    # json file with info about scenes
    #scenes_eval = json.load(open(cfg["scenes_names"]))
    #read all the .wav files in the enhanced folder
    scenes_eval = [os.path.splitext(f)[0] for f in os.listdir(enhanced) if f.endswith('.wav')]
    # csv file to store metrics
    create_dir('/home/jesus10/jrenato/avse/objective_evaluation/')
    metrics_file = os.path.join('/home/jesus10/jrenato/avse/objective_evaluation/', "metrics.csv")
    csv_lines = [["scene", "stoi", "pesq"]]
    
    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for scene in scenes_eval:
            futures.append(executor.submit(run_metrics, scene, enhanced, target))
        proc_list = [future.result() for future in tqdm(futures)]

    # Store results in one file
    with open(metrics_file, "w") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for scene_name in tqdm(scenes_eval):
            # scene_name = scene["scene"]
            scene_metrics_file = os.path.join('/home/jesus10/jrenato/avse/objective_evaluation/', f"{scene_name}.csv")
            with open(scene_metrics_file, newline='') as csv_f:
                scene_metrics = csv.reader(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in scene_metrics:
                    csv_writer.writerow(row)
            # remove tmp file
            os.system(f"rm {scene_metrics_file}")

if __name__ == "__main__":
    compute_metrics()