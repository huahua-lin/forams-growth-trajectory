# From Instance Segmentation to 3D Growth Trajectory Reconstruction in Planktonic Foraminifera
## 📦 Installation
1. **Clone the repository**
```
git clone https://github.com/huahua-lin/forams-growth-trajectory.git
```
2. **Create a new environment using Conda**
```
conda env create -f environment.yml  # create environment from YAML
conda activate foram  # activate the environment
```
## 🧪 Example Usage
### Instance Segmentation
1. Store your dataset under a folder.
2. Go to `instance_seg` folder and choose one pipeline.
3. If you choose `mtl_sw`:
   1. Run `train.py` for training the U-Net for semantic segmentation of chambers.
   2. Run `pred.py` for predicting the semantic segmentation. 
   3. Run `seeded_watershed.py` for instance segmentation.
### Trajectory Reconstruction
1. Go to `chamber_ordering` folder and run `nearest-neighbour.py`.

## 🔧 Contact
If you have any questions, do not hesitate to contact us by `huahua.lin@soton.ac.uk`.