### Landmarks Annotation Demo 1

📊 **Example Table Format**:


![Annotation Table](./documentation/images/DEMO1_table.png)

---

🛠️ **Update and change the config file**


```yaml
file_paths_csv: "/home/clemens/code/RA/landmarks_utils/documentation/DEMO_annotation/data_0_F_tr.csv"
save_dst_dir: "/home/clemens/code/RA/landmarks_utils/documentation/DEMO_annotation/out_F/"

LANDMARK_NAMES:
 - 'TMT1-D'
 - 'MTB1'
 - 'MTP1-P'
 - 'MTP1-D'
 - 'FIP1-P'
 - 'FIP1-D'
 - 'FDP1-P'
 - 'TMT2-D'
 - 'MTB2'
 - 'MTP2-P'
 - 'MTP2-D'
 - 'FIP2-P'
 - 'TMT3-D'
 - 'MTB3'
 - 'MTP3-P'
 - 'MTP3-D'
 - 'FIP3-P'
 - 'TMT4-D'
 - 'MTB4'
 - 'MTP4-P'
 - 'MTP4-D'
 - 'FIP4-P'
 - 'TMT5-D'
 - 'MTB5'
 - 'MTP5-P'
 - 'MTP5-D'
 - 'FIP5-P'


## The base path in the table (specified in file_paths_csv) was set to a different computer. One can either change it in the table or use the "re-rooting" functionality below
reroot_file_path_dir__from: "/home/cwatzenboeck/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms/"
reroot_file_path_dir__to: "/home/clemens/data/AutoPIX_cirdata/projects__autoscora/autoscoRA_images/F_images_of_interest_2_renamed_mirrored_inverted_dicoms/"
```


It can simply be run as: 

```bash 
   landmarks_utils__annotate_landmarks --config config_H.yml 
```


###### DEMO

![Annotation Demo](./documentation/images/DEMO_annotation_v1_compressed.gif)



