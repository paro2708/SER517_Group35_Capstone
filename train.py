from GazeRefineNet import GRN
import pytorch_lightning as pl

if __name__ == '__main__':
    
    # image_dir = r'C:\Program Files\Common Files\ProDataset\train\images\cropped_eyes'
    # meta_dir = r'C:\Program Files\Common Files\ProDataset\train\meta'
    save_dir = "../Result"

    image_dir = r'C:\Users\ASU Zoom 01\Downloads\ProDataset\train\images\iPhone 5S\cropped_eyes'
    meta_dir= r'C:\Users\ASU Zoom 01\Downloads\ProDataset\train\meta'

    model = GRN(data_path= image_dir, save_path= save_dir)

    trainer = pl.Trainer(accelerator= "cpu", max_epochs= 5, default_root_dir= save_dir, enable_progress_bar= True)
    trainer.fit(model)

    print("Done")