import soundfile as sf
import pandas as pd
from pathlib import Path
from xml.dom import minidom
from tqdm import tqdm

def get_annotation(file_name, annotations_path):
    # Process the .svl xml file
    svl_path = str(Path(annotations_path, file_name + ".svl"))
    xmldoc = minidom.parse(svl_path)
    itemlist = xmldoc.getElementsByTagName("point")

    start_time = []
    end_time = []
    labels = []

    if len(itemlist) > 0:
        for s in itemlist:
            start_frame = float(s.attributes["frame"].value)
            label = str(s.attributes["label"].value)
            label_string = label.split(",")[0] if "," in label else label

            if label == "":
                continue

            annotation_duration = float(s.attributes["duration"].value)
            start_time.append(start_frame)
            end_time.append(start_frame + annotation_duration)
            labels.append(label_string)

    df_annotations = pd.DataFrame({"Start": start_time, "End": end_time, "Label": labels})
    return df_annotations

def process_audio_file(file_name, base_path):
    annotations_path = Path(base_path, "Annotations")
    audio_path = Path(base_path, "Audio")
    processed_audio_path = Path(base_path, "ProcessedAudio")
    processed_audio_path.mkdir(exist_ok=True)

    annotations = get_annotation(file_name, annotations_path)
    audio_file = audio_path / (file_name + ".wav")

    file_info = []
    with sf.SoundFile(str(audio_file)) as audio_data:
        sample_rate = audio_data.samplerate
        total_frames = audio_data.frames

        for index, row in annotations.iterrows():
            start_frame = int(row['Start'])
            end_frame = int(row['End'])

            # Ensure frame bounds are within the audio file length
            start_frame = max(0, min(start_frame, total_frames))
            end_frame = max(0, min(end_frame, total_frames))

            if start_frame < end_frame:
                audio_data.seek(start_frame)
                audio_part = audio_data.read(frames=end_frame - start_frame)

                new_file_name = f"{file_name}_part{index}.wav"
                new_file_path = str(processed_audio_path / new_file_name)
                sf.write(new_file_path, audio_part, sample_rate)
                file_info.append({'IN FILE': new_file_name, 'MANUAL ID': row['Label']})
            else:
                print(f"Skipping invalid frame range for {file_name}, part {index}")

    return file_info

def process_all_audio_files(base_path):
    annotations_path = Path(base_path, "Annotations")
    all_files_info = []

    for annotation_file in tqdm(list(annotations_path.glob("*.svl")), desc="Processing Files"):
        file_name = annotation_file.stem
        file_info = process_audio_file(file_name, base_path)
        all_files_info.extend(file_info)

    df_all_files = pd.DataFrame(all_files_info)
    df_all_files.to_csv(str(Path(base_path, "meta.csv")), index=False, sep=';')

# Example usage
process_all_audio_files("/pfs/work7/workspace/scratch/ul_xto11-blah/")
