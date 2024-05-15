# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:25:11 2024

@author: PHY3BOHORT
"""

########################
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import glob
import os ,sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import gc
import matplotlib.pyplot as plt
import shutil
import tkinter.ttk as ttk
# import pyautogui
from PIL import ImageTk, Image
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from tkinter import messagebox
from skimage import io, transform
import pandas as pd
from datetime import datetime
from tkinter import PhotoImage
import keras.backend as K
import pathlib
from tqdm import tqdm
# import tensorflow_addons as tfa
# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# except:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################

# import ctypes, tkinter
# try: # >= win 8.1
#     ctypes.windll.shcore.SetProcessDpiAwareness(2)
# except: # win 8.0 or less
#     ctypes.windll.user32.SetProcessDPIAware()




root = tk.Tk()
root.title('SATORI V1.8.0++')

# root.state('zoomed') # Docker _tkinter.TclError: bad argument "zoomed": must be normal, iconic, or withdrawn

# root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.geometry("{0}x{1}+0+0".format(1920,1000))


# curr_width, curr_height= pyautogui.size()
# print(curr_width, curr_height)
# gui_width = 1920
# gui_height =1080

# wd_scale = curr_width-1920
# ht_scale = curr_height-1080
# print(wd_scale, ht_scale)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)
root.columnconfigure(3, weight=1)
root.columnconfigure(4, weight=1)
root.columnconfigure(5, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)


global W
global H
global slices
global pixel_area
global Filter_Key
global count_files
global save_folder_name
Filter_Key = tk.StringVar()
Filter_Key.set("")


# if os.path.exists('./result'):
#     shutil.rmtree('./result')

# try:
#     if os.path.exists('./result'):
#         os.remove('./result')
#     os.mkdir('./result')
# except:
#     pass

global AAo_arr
global DAo_arr
AAo = []
DAo = []
global image_on_canvas
global new_on_canvas
global mode_var
#########################################################################


def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint8)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint8)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint8)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint8)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)






ALPHA = 0.8
BETA = 0.8
GAMMA = 1

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def position_cnt(a):
    momA = cv2.moments(a)
    (xa,_) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    return xa

def assign_lbl(image,disp = False):
    img = image.copy()

    img = img.astype(np.uint8)
    img = cv2.UMat(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt=[]
    # cnt_area =[]
    cnt_area_pairs = []
    for i in range(len(contours)):

        area_chk = cv2.contourArea(contours[i])


        if area_chk > 5:
            # cnt.append(contours[i])
            # cnt_area.append(int(cv2.contourArea(contours[i])))
            cnt_area_pairs.append((contours[i], int(cv2.contourArea(contours[i]))))
    
    cnt_area_pairs.sort(key=lambda x: x[1], reverse=True)
    cnt = [pair[0] for pair in cnt_area_pairs]
    cnt_area = [pair[1] for pair in cnt_area_pairs]
    
    cnt = cnt[:2]
    cnt_area = cnt_area[:2]
    
    # print(cnt_area)
    
    
    if len(cnt_area)>=2:
        cnt_area.sort(reverse = True)

    try:
        co1 = position_cnt(cnt[0])
    except:
        # print("debug1:co1",position_cnt(cnt[0]))
        pass

    try:
        co2 = position_cnt(cnt[1])
    except:
        pass


    try:
            
        if cnt_area[0] > cnt_area[1]:
            AAo.append(int(cnt_area[0]))
            DAo.append(int(cnt_area[1]))
        else:
            AAo.append(int(cnt_area[1]))
            DAo.append(int(cnt_area[0]))
    except:
            AAo.append(int(0))
            DAo.append(int(0))
            
    

    try:
        # po1 = 128 - co1
        po1 = (co1 + co2) // 2
        
    except:
        po1 = 128 - co1
        

    # try:
    #     po2 = 130 - co2
    # except:
    #     pass

    if po1 > 0:
        try:
            img = cv2.drawContours(img, cnt, 0, (1),cv2.FILLED)
        except:
            pass
        try:
            img = cv2.drawContours(img, cnt, 1, (2),cv2.FILLED)
        except:
            pass
        
    elif po1 < 0 :
        try:
            img = cv2.drawContours(img, cnt, 1, (1),cv2.FILLED)
        except:
            pass
        try:
            img = cv2.drawContours(img, cnt, 0, (2),cv2.FILLED)
        except:
            pass
    img = cv2.UMat.get(img)
    return img


model =load_model(r"proposedmodel.hdf5" ,custom_objects={"FocalTverskyLoss":FocalTverskyLoss, "dice_coef_loss": dice_coef_loss ,"dice_coef":dice_coef})

def save_img(fr_img,output_img,path,i):
    global img_W, img_H, rotation_angle

##    fr_img = np.rot90(fr_img)
##    fr_img = np.rot90(fr_img)
##
##    output_img = np.rot90(output_img)
##    output_img = np.rot90(output_img)


    output_img = np.ma.masked_where(output_img < 0.5, output_img)
    output_img[i][output_img[i] <= 2] = 0
    for j in range(output_img.shape[0]):
        for k in range(output_img.shape[1]):
            if output_img[j,k] == 1:
                output_img[j,k] =2
            elif output_img[j,k] == 2:
                output_img[j,k] =1

    # from skimage.transform import resize
    # import io
    plt.imshow(fr_img,cmap="gray")
    plt.imshow(output_img,cmap="jet",interpolation='none', alpha=0.5)
    plt.axis('off')
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)

    # Convert the buffer to a NumPy array
    # image = np.array(Image.open(buf))
    # new_shape = (img_H, img_W)
    # resized_image = resize(image, new_shape, anti_aliasing=False)
    # plt.imshow(resized_image)
    plt.savefig(path, dpi=300,bbox_inches='tight',pad_inches = 0)
    plt.close('all')

#########################################################################      open directory
def open_dir():

    lb.state(('disabled',))
    deactivate_button()
    global image_path
    global image_path_unique
    global Filter_Key
    update_canvas_with_image('logo.png')
    # Reset Treeview
    for i in lb.get_children():
        lb.delete(i)
    image_path = filedialog.askdirectory(title="Select a Directory" )
    image_path_unique = image_path

    if not image_path:  # Exit if the user cancels.
        activate_button()
        return
    f1 = glob.glob(os.path.join(image_path, "*"))

    photo_images = []  # Keep references to PhotoImages

    progbar = ttk.Progressbar(root,  orient=tk.HORIZONTAL, length=800, mode='determinate')
    progbar.grid(column=1,row=5,rowspan=2,sticky="w")
    global idx
    idx = 1
    for f in f1:

        progbar['value'] = (int(f1.index(f)+1)/int(len(f1)))*100
        progbar.update_idletasks()
        progbar.update()
        root.update()

        if f.lower().endswith(('ima', 'nii', 'dcm')):
            if f.endswith("nii"):
                header = nib.load(f).header

                if Filter_Key.get() == "":
                    image = nib.load(f).get_fdata()[:, :, 0]
                    image = Image.fromarray(image)
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference

                elif all(word in str(header).lower() for word in Filter_Key.get().lower().split()):
##                elif Filter_Key.get().lower() in str(header).lower():
                    image = nib.load(f).get_fdata()[:, :, 0]
                    image = Image.fromarray(image)
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference
            else:
                dicom_data = pydicom.dcmread(f)


                if Filter_Key.get() == "":

                    image_array = dicom_data.pixel_array

                    if len(image_array.shape)>=3:
                        image_array = image_array[:, :, 0]
                    else:
                        pass

                    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                    # Avoid division by zero by ensuring the maximum value is greater than zero
                    max_val = np.max(image_array)
                    if max_val > 0:
                        # Normalize the image data to 0-255
                        image_array = (image_array / max_val * 255).astype(np.uint8)
                    else:
                        # Handle the case where the image is completely black or max_val is 0
                        image_array = np.zeros(image_array.shape, dtype=np.uint8)

                    # Convert the numpy array to a PIL Image
                    image = Image.fromarray(image_array)
                    if image.mode != 'L':
                        image = image.convert('L')  # Convert to grayscale if not already
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference




##                        if all(word in str(Description).lower() for word in Filter_Key.get().lower().split()):

                elif Filter_Key.get().lower() in str(dicom_data).lower():
                    image_array = dicom_data.pixel_array

                    if len(image_array.shape)>=3:
                        image_array = image_array[:, :, 0]
                    else:
                        pass

                    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                    # Avoid division by zero by ensuring the maximum value is greater than zero
                    max_val = np.max(image_array)
                    if max_val > 0:
                        # Normalize the image data to 0-255
                        image_array = (image_array / max_val * 255).astype(np.uint8)
                    else:
                        # Handle the case where the image is completely black or max_val is 0
                        image_array = np.zeros(image_array.shape, dtype=np.uint8)

                    # Convert the numpy array to a PIL Image
                    image = Image.fromarray(image_array)
                    if image.mode != 'L':
                        image = image.convert('L')  # Convert to grayscale if not already
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference

        idx = idx+1


    lb.photo_images = photo_images  # Attach to the Treeview widget to keep alive
    if lb.get_children():
        lb.selection_set(lb.get_children()[0])  # Select the first item
        lb.event_generate("<<TreeviewSelect>>")
    btn_go.config(state="normal")
    progbar.destroy()
    activate_button()
    count_files.set('Total Number of Files: '+str(len(lb.get_children())))
    lb.state(('!disabled',))
    gc.collect()
#########################################################################      Filter
def Filter_Files():
    global image_path
    global Filter_Key
    # Reset Treeview
    for i in lb.get_children():
        lb.delete(i)

    update_canvas_with_image('logo.png')
    deactivate_button()
    lb.state(('disabled',))
    # image_path = filedialog.askdirectory()
    # if not image_path:  # Exit if the user cancels.
    #     return
    f1 = glob.glob(os.path.join(image_path, "*"))

    photo_images = []  # Keep references to PhotoImages

    progbar = ttk.Progressbar(root,  orient=tk.HORIZONTAL, length=800, mode='determinate')
    progbar.grid(column=1,row=5,rowspan=2,sticky="w")
    global idx
    idx = 1
    for f in f1:
        print(idx)
        progbar['value'] = (int(f1.index(f)+1)/int(len(f1)))*100
        progbar.update_idletasks()
        progbar.update()
        root.update()
        if f.lower().endswith(('ima', 'nii', 'dcm')):
            if f.endswith("nii"):
                header = nib.load(f).header

                if Filter_Key.get() == "":
                    image = nib.load(f).get_fdata()[:, :, 0]
                    image = Image.fromarray(image)
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference
##                elif all(word in str(header).lower() for word in Filter_Key.get().lower().split()):
                elif Filter_Key.get().lower() in str(header) or  Filter_Key.get().lower() in f.lower():
                    image = nib.load(f).get_fdata()[:, :, 0]
                    image = Image.fromarray(image)
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference
            else:
                dicom_data = pydicom.dcmread(f)


                if Filter_Key.get() == "":

                    image_array = dicom_data.pixel_array

                    if len(image_array.shape)>=3:
                        image_array = image_array[:, :, 0]
                    else:
                        pass

                    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                    # Avoid division by zero by ensuring the maximum value is greater than zero
                    max_val = np.max(image_array)
                    if max_val > 0:
                        # Normalize the image data to 0-255
                        image_array = (image_array / max_val * 255).astype(np.uint8)
                    else:
                        # Handle the case where the image is completely black or max_val is 0
                        image_array = np.zeros(image_array.shape, dtype=np.uint8)

                    # Convert the numpy array to a PIL Image
                    image = Image.fromarray(image_array)
                    if image.mode != 'L':
                        image = image.convert('L')  # Convert to grayscale if not already
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference

##                elif all(word in str(dicom_data).lower() for word in Filter_Key.get().lower().split()):
                elif Filter_Key.get().lower() in str(dicom_data).lower() or  Filter_Key.get().lower() in f.lower():
                    image_array = dicom_data.pixel_array

                    if len(image_array.shape)>=3:
                        image_array = image_array[:, :, 0]
                    else:
                        pass

                    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                    # Avoid division by zero by ensuring the maximum value is greater than zero
                    max_val = np.max(image_array)
                    if max_val > 0:
                        # Normalize the image data to 0-255
                        image_array = (image_array / max_val * 255).astype(np.uint8)
                    else:
                        # Handle the case where the image is completely black or max_val is 0
                        image_array = np.zeros(image_array.shape, dtype=np.uint8)

                    # Convert the numpy array to a PIL Image
                    image = Image.fromarray(image_array)
                    if image.mode != 'L':
                        image = image.convert('L')  # Convert to grayscale if not already
                    image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                    photo = ImageTk.PhotoImage(image)
                    lb.insert('', 'end', values=(str(idx),os.path.basename(f)), image=photo)

                    photo_images.append(photo)  # Store reference
        idx=idx+1

    lb.photo_images = photo_images  # Attach to the Treeview widget to keep alive
    if lb.get_children():
        lb.selection_set(lb.get_children()[0])  # Select the first item
        lb.event_generate("<<TreeviewSelect>>")
    btn_go.config(state="normal")
    progbar.destroy()
    activate_button()
    count_files.set('Total Number of Files: '+str(len(lb.get_children())))
    lb.state(('!disabled',))
    gc.collect()



def copy_images():
    try:
        predicting_imge = os.path.join(image_path,str(select_img.get()))
    except:
        tk.messagebox.showwarning("[info]", "Select an Image")
        return
    save_path = filedialog.askdirectory(title="Save Directory")

    if not save_path:  # Exit if the user cancels.
        save_path  = os.getcwd()
        return



    if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)),r'./result/psudo_segmentation.npy')):
        pred_lbl = os.path.join(os.path.dirname(os.path.abspath(__file__)),r'./result/psudo_segmentation.npy')
        pred_np = np.load(pred_lbl)
        # print(pred_np.shape)

        path_save = os.path.join(save_path, os.path.basename(predicting_imge))+str(".npy")
        # print(path_save)
        np.save(path_save, pred_np)

    try:
        shutil.copy(predicting_imge, save_path)

    except:
        pass



    tk.messagebox.showwarning("[info]", "Done!")
def callback(event):
    global rotation_angle
    rotation_angle=0
    # if os.path.exists('./result'):
    #     shutil.rmtree('./result')
    selected_item = event.widget.selection()
    if selected_item:  # Ensure there's a selection
        
        item = selected_item[0]  # Get the first (or only) selected item
        item_values = event.widget.item(item, 'values')  # Get the text of the item

        select_img.set(str(os.path.basename(item_values[1]).replace("\\","/")))

        if item_values[1].endswith(".nii"):
            select_nii()  # Assuming this is another function you've defined
        else:
            select_dicom()
    else:
        select_img.set("")

def mass_Filter():

    window_filter = tk.Toplevel(root)
    window_filter.title("[Mass Filter and Copy]")
    window_filter.geometry("{0}x{1}+0+0".format(1920,1000))


    open_dir_path1 =tk.StringVar()
    open_dir_path1.set("Open Directory : ")


    save_dir_path1 =tk.StringVar()
    save_dir_path1.set("Save Directory : ")

    f1=None
    image_path=None
    f_Filter_Key = tk.StringVar()
    f_Filter_Key.set("")


    global f_count_files

    f_count_files = tk.StringVar()
    f_count_files.set('Total Number of Files: 0')

    global photo_lbl
    global f_image_path
    f_image_path=None
    def close_window():
        window_filter.destroy()


    # Function to update the label image
    def update_image(index):
        global image_sequence
        # Update the image of the label
        photo = PhotoImage(file=image_sequence[index])
        label.config(image=photo)
        label.image = photo  # Keep a reference!
        # Schedule the next image update
        next_index = (index + 1) % len(image_sequence)
        root.after(100, update_image, next_index)  # Adjust delay as needed


    # Create a frame for the background
    background_frame = tk.Frame(window_filter, bg='lightgray', width=250, height=250)
    background_frame.grid(column=0, row=3, columnspan=1, sticky="")
    background_frame.grid_propagate(False)

    # Global variable for the label widget that will display images
    global ckbox_label_image
    ckbox_label_image = tk.Label(background_frame, bg='lightgray')
    ckbox_label_image.place(relx=0.5, rely=0.5, anchor="center")



    def f_callback(event):
        global f_image_path
        global photo_lbl

        selected_item = event.widget.selection()
        if selected_item:  # Ensure there's a selection
            item = selected_item[0]  # Get the first (or only) selected item
            item_values = event.widget.item(item, 'values')  # Get the text of the item

            f = os.path.join(f_image_path, item_values[1])

            if item_values[1].endswith(".nii"):
                image = nib.load(f).get_fdata()[:, :, 0]
                image = Image.fromarray(image)
                image.thumbnail((260, 250))  # Adjust thumbnail size as needed
                photo = ImageTk.PhotoImage(image)
                ckbox_label_image.config(image=photo)  # Update the label with the new image
                ckbox_label_image.image = photo

            else:
                dicom_data = pydicom.dcmread(f)

                image_array = dicom_data.pixel_array

                if len(image_array.shape)>=3:
                    image_array = image_array[:, :, 0]
                else:
                    pass

                image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                # Avoid division by zero by ensuring the maximum value is greater than zero
                max_val = np.max(image_array)
                if max_val > 0:
                    # Normalize the image data to 0-255
                    image_array = (image_array / max_val * 255).astype(np.uint8)
                else:
                    # Handle the case where the image is completely black or max_val is 0
                    image_array = np.zeros(image_array.shape, dtype=np.uint8)

                # Convert the numpy array to a PIL Image
                image = Image.fromarray(image_array)
                if image.mode != 'L':
                    image = image.convert('L')  # Convert to grayscale if not already
                image.thumbnail((250, 250))  # Adjust thumbnail size as needed
                photo = ImageTk.PhotoImage(image)


                ckbox_label_image.config(image=photo)  # Update the label with the new image
                ckbox_label_image.image = photo


            return item_values[1]


    def f_open_dir():
        global f_count_files
        global f_image_path
        global file_images
        
        
        def read_folder_names_from_excel(excel_path):
            df = pd.read_excel(excel_path, engine='openpyxl')
            return df['Folder Name'].unique()
        
        try:
            f_image_path = filedialog.askdirectory(title="Select a Directory")
            folder_names = read_folder_names_from_excel('./log_file.xlsx')
            if any(pt in f_image_path for pt in folder_names):
                result = messagebox.askyesno("[info:]", "The file has been processed. Do you want to proceed?")
                if result:
                    pass
                else:
                    f_image_path = filedialog.askdirectory(title="Select a Directory")
            
            
            open_dir_path1.set("Open Directory : "+str(f_image_path))
            open_dir_path1.set(truncate_path(open_dir_path1.get()))
        except:
            pass
        window_filter.lift()
        # Reset Treeview
        for i in f_lb.get_children():
            f_lb.delete(i)



        f1 = glob.glob(os.path.join(f_image_path, "*"))

        photo_images = []  # Keep references to PhotoImages
        file_images = []
        try:

            progbar = ttk.Progressbar(window_filter,  orient=tk.HORIZONTAL, length=800, mode='determinate')
            progbar.grid(column=1,row=6,columnspan=2,sticky="new")
            global idx
            idx = 1
            for f in f1:
                progbar['value'] = (int(f1.index(f)+1)/int(len(f1)))*100
                progbar.update_idletasks()
                progbar.update()
                window_filter.update()

                if f.lower().endswith(('ima', 'nii', 'dcm')):
                    if f.endswith("nii"):
                        header = nib.load(f).header

                        if f_Filter_Key.get() == "":
                            image = nib.load(f).get_fdata()[:, :, 0]
                            image = Image.fromarray(image)
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f),"" , "[ ]"), image=photo)


                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                        elif f_Filter_Key.get().lower() in str(header).lower():
                            image = nib.load(f).get_fdata()[:, :, 0]
                            image = Image.fromarray(image)
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f),"" ,"[ ]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                    else:
                        dicom_data = pydicom.dcmread(f)
                        
                        try:
                            Description = dicom_data.SeriesDescription
                        except:
                            Description = ""
                        
                        # if dicom_data.SeriesDescription:
                        #     Description = dicom_data.SeriesDescription
                        # else:
                        #     Description = ""

                        if f_Filter_Key.get() == "":

                            image_array = dicom_data.pixel_array

                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass

                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)

                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)

    ##                        f_lb.insert('', 'end', values=os.path.basename(f), image=photo)
    ##                        f_lb.insert("", "end", values=(photo, os.path.basename(f), "[ ]"))
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f), Description ,"[ ]"), image=photo)

                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference

                        elif f_Filter_Key.get().lower() in str(dicom_data).lower():
                            image_array = dicom_data.pixel_array

                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass

                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)

                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f),Description , "[ ]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                idx = idx+1

        except:

            pass
        progbar.destroy()
        window_filter.update()
        f_count_files.set('Total Number of Files: '+str(len(f_lb.get_children())))
        f_lb.photo_images = photo_images  # Attach to the Treeview widget to keep alive
        if f_lb.get_children():
            f_lb.selection_set(f_lb.get_children()[0])  # Select the first item
            f_lb.event_generate("<<TreeviewSelect>>")
            
            
    def f_open_dir_next_prev(f_image):
        global f_count_files
        global f_image_path
        global file_images
        f_image_path = f_image
        
        def read_folder_names_from_excel(excel_path):
            df = pd.read_excel(excel_path, engine='openpyxl')
            return df['Folder Name'].unique()
        
        # try:
        #     f_image_path = filedialog.askdirectory(title="Select a Directory")
        #     folder_names = read_folder_names_from_excel('./log_file.xlsx')
        #     if any(pt in f_image_path for pt in folder_names):
        #         result = messagebox.askyesno("[info:]", "The file has been processed. Do you want to proceed?")
        #         if result:
        #             pass
        #         else:
        #             f_image_path = filedialog.askdirectory(title="Select a Directory")
            
            
        #     open_dir_path1.set("Open Directory : "+str(f_image_path))
        #     open_dir_path1.set(truncate_path(open_dir_path1.get()))
        # except:
        #     pass
        window_filter.lift()
        # Reset Treeview
        
        
        for i in f_lb.get_children():
            f_lb.delete(i)



        f1 = glob.glob(os.path.join(f_image_path, "*"))

        photo_images = []  # Keep references to PhotoImages
        file_images = []
        try:

            progbar = ttk.Progressbar(window_filter,  orient=tk.HORIZONTAL, length=800, mode='determinate')
            progbar.grid(column=1,row=6,columnspan=2,sticky="new")
            global idx
            idx = 1
            for f in f1:
                progbar['value'] = (int(f1.index(f)+1)/int(len(f1)))*100
                progbar.update_idletasks()
                progbar.update()
                window_filter.update()

                if f.lower().endswith(('ima', 'nii', 'dcm')):
                    if f.endswith("nii"):
                        header = nib.load(f).header

                        if f_Filter_Key.get() == "":
                            image = nib.load(f).get_fdata()[:, :, 0]
                            image = Image.fromarray(image)
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert('', 'end', values=(idx ,os.path.basename(f),"" , "[ ]"), image=photo)
                            

                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                        elif f_Filter_Key.get().lower() in str(header).lower() or f_Filter_Key.get().lower() in str(os.path.basename(f)):
                            image = nib.load(f).get_fdata()[:, :, 0]
                            image = Image.fromarray(image)
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert(idx,'', 'end', values=(os.path.basename(f),"" ,"[ ]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                    else:
                        dicom_data = pydicom.dcmread(f)
                        
                        try:
                            Description = dicom_data.SeriesDescription
                        except:
                            Description = ""
                        
                        # if dicom_data.SeriesDescription:
                        #     Description = dicom_data.SeriesDescription
                        # else:
                        #     Description = ""

                        if f_Filter_Key.get() == "":

                            image_array = dicom_data.pixel_array

                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass

                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)

                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)

    ##                        f_lb.insert('', 'end', values=os.path.basename(f), image=photo)
    ##                        f_lb.insert("", "end", values=(photo, os.path.basename(f), "[ ]"))
                            f_lb.insert(idx,'', 'end', values=(os.path.basename(f), Description ,"[ ]"), image=photo)

                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference

                        elif f_Filter_Key.get().lower() in str(dicom_data).lower() or f_Filter_Key.get().lower() in str(os.path.basename(f)):
                            image_array = dicom_data.pixel_array

                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass

                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)

                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
                            f_lb.insert(idx,'', 'end', values=(os.path.basename(f),Description , "[ ]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                idx = idx+1

        except:

            pass
        progbar.destroy()
        window_filter.update()
        f_count_files.set('Total Number of Files: '+str(len(f_lb.get_children())))
        f_lb.photo_images = photo_images  # Attach to the Treeview widget to keep alive
        if f_lb.get_children():
            f_lb.selection_set(f_lb.get_children()[0])  # Select the first item
            f_lb.event_generate("<<TreeviewSelect>>")

    def f_search_dir():
        global f_image_path
        global f_count_files
        global file_images
        window_filter.lift()

        # if f_image_path == None:
        #     tk.messagebox.showwarning("[info]", "Select a Directory")
        #     window_filter.lift()
        #     return
        # window_filter.lift()
        # Reset Treeview

        try:
            f1 = glob.glob(os.path.join(f_image_path, "*"))
        except:
            tk.messagebox.showwarning("[info]", "Select a Directory")
            window_filter.lift()
            return



        filter_list = []
        for item in f_lb.get_children():
            values = list(f_lb.item(item, "values"))
            if values[3] == "[x]":
                filter_list.append(values[1])
            f_lb.delete(item)




        photo_images = []  # Keep references to PhotoImages
        file_images = []
        progbar = ttk.Progressbar(window_filter,  orient=tk.HORIZONTAL, length=800, mode='determinate')
        progbar.grid(column=1,row=6,columnspan=2,sticky="new")
        global idx
        idx = 1
        for f in f1:
            progbar['value'] = (int(f1.index(f)+1)/int(len(f1)))*100
            progbar.update_idletasks()
            progbar.update()
            window_filter.update()
            if f.lower().endswith(('ima', 'nii', 'dcm')):
                if f.endswith("nii"):
                    header = nib.load(f).header

                    if f_Filter_Key.get() == "":
                        image = nib.load(f).get_fdata()[:, :, 0]
                        image = Image.fromarray(image)
                        image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                        photo = ImageTk.PhotoImage(image)
                        if len(filter_list)>=1:
                            if any(os.path.basename(f) in s for s in filter_list):
                                f_lb.insert('', 'end', values=(idx,os.path.basename(f),"" ,"[x]"), image=photo)
                                file_images.append(os.path.basename(f))
                                photo_images.append(photo)  # Store reference
                        else:

                            if any(os.path.basename(f) in s for s in filter_list):
                                f_lb.insert('', 'end', values=(idx,os.path.basename(f),"" ,"[x]"), image=photo)
                                file_images.append(os.path.basename(f))
                                photo_images.append(photo)  # Store reference
                            else:
                                f_lb.insert('', 'end', values=(idx,os.path.basename(f),"" ,"[ ]"), image=photo)

                                file_images.append(os.path.basename(f))
                                photo_images.append(photo)  # Store reference

##                        file_images.append(os.path.basename(f))
##                        photo_images.append(photo)  # Store reference


                    elif f_Filter_Key.get().lower() in str(header).lower() or any(os.path.basename(f) in s for s in filter_list) or f_Filter_Key.get().lower() in str(os.path.basename(f)).lower():
                        image = nib.load(f).get_fdata()[:, :, 0]
                        image = Image.fromarray(image)
                        image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                        photo = ImageTk.PhotoImage(image)
                        if any(os.path.basename(f) in s for s in filter_list):
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f),"", "[x]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference
                        else:
                            f_lb.insert('', 'end', values=(idx,os.path.basename(f),"", "[ ]"), image=photo)
                            file_images.append(os.path.basename(f))
                            photo_images.append(photo)  # Store reference

##                        file_images.append(os.path.basename(f))
##                        photo_images.append(photo)  # Store reference
                else:
                    try:
                        dicom_data = pydicom.dcmread(f)
                        
                        # dicom_data = pydicom.dcmread(f)
                    
                        try:
                            Description = dicom_data.SeriesDescription
                        except:
                            Description = ""
                            
                        # if dicom_data.SeriesDescription:
                        #     Description = dicom_data.SeriesDescription
                        # else:
                        #     Description = ""
    
    
                        if f_Filter_Key.get() == "":
    
                            image_array = dicom_data.pixel_array
    
                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass
    
                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))
    
                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)
    
                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
    
                            if len(filter_list)>=1:
                                if any(os.path.basename(f) in s for s in filter_list):
                                    f_lb.insert('', 'end', values=(idx,os.path.basename(f),Description , "[x]"), image=photo)
                                    file_images.append(os.path.basename(f))
                                    photo_images.append(photo)  # Store reference
                            else:
    
                                if any(os.path.basename(f) in s for s in filter_list):
                                    f_lb.insert('', 'end', values=(idx,os.path.basename(f),Description , "[x]"), image=photo)
                                    file_images.append(os.path.basename(f))
                                    photo_images.append(photo)  # Store reference
                                else:
                                    f_lb.insert('', 'end', values=(idx,os.path.basename(f),Description , "[ ]"), image=photo)
    
                                    file_images.append(os.path.basename(f))
                                    photo_images.append(photo)  # Store reference
    
                        elif f_Filter_Key.get().lower() in str(dicom_data).lower() or any(os.path.basename(f) in s for s in filter_list) or f_Filter_Key.get().lower() in str(os.path.basename(f)):
                            image_array = dicom_data.pixel_array
    
                            if len(image_array.shape)>=3:
                                image_array = image_array[:, :, 0]
                            else:
                                pass
    
                            image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))
    
                            # Avoid division by zero by ensuring the maximum value is greater than zero
                            max_val = np.max(image_array)
                            if max_val > 0:
                                # Normalize the image data to 0-255
                                image_array = (image_array / max_val * 255).astype(np.uint8)
                            else:
                                # Handle the case where the image is completely black or max_val is 0
                                image_array = np.zeros(image_array.shape, dtype=np.uint8)
    
                            # Convert the numpy array to a PIL Image
                            image = Image.fromarray(image_array)
                            if image.mode != 'L':
                                image = image.convert('L')  # Convert to grayscale if not already
                            image.thumbnail((60, 60))  # Adjust thumbnail size as needed
                            photo = ImageTk.PhotoImage(image)
    
                            if any(os.path.basename(f) in s for s in filter_list):
                                f_lb.insert('', 'end', values=(idx,os.path.basename(f),Description , "[x]"), image=photo)
                                file_images.append(os.path.basename(f))
                                photo_images.append(photo)  # Store reference
                            else:
                                f_lb.insert('', 'end', values=(idx,os.path.basename(f), Description , "[ ]"), image=photo)
                                file_images.append(os.path.basename(f))
                                photo_images.append(photo)  # Store reference
                    except:
                        pass
                    
                    

##                        file_images.append(os.path.basename(f))
##                        photo_images.append(photo)  # Store reference
        progbar.destroy()
        window_filter.update()
        f_count_files.set('Total Number of Files: '+str(len(f_lb.get_children())))
        f_lb.photo_images = photo_images  # Attach to the Treeview widget to keep alive
        if f_lb.get_children():
            f_lb.selection_set(f_lb.get_children()[0])  # Select the first item
            f_lb.event_generate("<<TreeviewSelect>>")

    # Function to simulate checkbox toggle
    def toggle_checkbox(item):
##        item = f_lb.identify_row(event.y)
        if not item:
            return
        values = list(f_lb.item(item, "values"))

        if values[3] == "[ ]":
            values[3] = "[x]"
        else:
            values[3] = "[ ]"
        f_lb.item(item, values=values)



    def comp_move():
        # Function to append log to an Excel file
        def append_log_to_excel(log_file_path, folder_name, file_name):
            # Define the log entry
            log_entry = {
                'Date': [datetime.now()],
                'Folder Name': [folder_name],
                'File Name': [file_name]
            }

            # Convert log entry to DataFrame
            df_log_entry = pd.DataFrame(log_entry)

            # Check if the Excel log file exists
            if os.path.exists(log_file_path):
                # Read the existing data
                df_existing = pd.read_excel(log_file_path)
                # Append new log entry
                df_new = pd.concat([df_existing, df_log_entry], ignore_index=True)
            else:
                # If the file does not exist, start a new DataFrame
                df_new = df_log_entry

            # Write/overwrite the Excel file
            df_new.to_excel(log_file_path, index=False, engine='openpyxl')




        global f_image_path, f_save_image_path

        if f_save_image_path==None:
            tk.messagebox.showwarning("[info]", "Select a Directory")
            window_filter.lift()
            return

        if f_image_path==None:
            tk.messagebox.showwarning("[info]", "Select a Directory")
            window_filter.lift()
            return


        concatenated_images = None
        affine = np.eye(4)  # Default to identity matrix

        # Setup progress bar
        progbar = ttk.Progressbar(window_filter, orient=tk.HORIZONTAL, length=800, mode='determinate')
        progbar.grid(column=1, row=6, columnspan=2, sticky="new")

        # Iterate over selected items
        for index, item in enumerate(f_lb.get_children()):
            # Update progress bar
            progbar['value'] = (index + 1) / len(f_lb.get_children()) * 100
            progbar.update_idletasks()

            # Process item if selected
            values = list(f_lb.item(item, "values"))
            filename, checkbox_indicator = values[1], values[3]
            if checkbox_indicator == "[x]":
                full_path = os.path.join(f_image_path, filename)
                if filename.lower().endswith("nii"):
                    nii = nib.load(full_path)
                    image = nii.get_fdata()
                    if concatenated_images is None:
                        affine = nii.affine  # Use the affine of the first NIfTI image
                else:  # Assume DICOM
                    try:
                        dicom_data = pydicom.dcmread(full_path, force=True)
                        image = dicom_data.pixel_array
                        if concatenated_images is None:
                            # Simplified affine calculation for DICOM, based on pixel spacing and slice thickness
                            pixel_spacing = dicom_data.PixelSpacing
                            slice_thickness = dicom_data.SliceThickness
                            affine[0,0], affine[1,1], affine[2,2] = pixel_spacing[0], pixel_spacing[1], slice_thickness
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
                        continue

                # Initialize or concatenate images
                image = np.expand_dims(image, axis=-1)  # Ensure at least 4D for easy concatenation
                concatenated_images = image if concatenated_images is None else np.concatenate((concatenated_images, image), axis=-1)

        # Cleanup
        progbar.destroy()

        # Cleanup
        progbar.destroy()
        window_filter.update()

        # Save concatenated image as DICOM
        if concatenated_images is not None:
            nifti_img = nib.Nifti1Image(concatenated_images.squeeze(), affine)

            # Save the DICOM file
            file_name_from_firs_file = str(os.path.basename(f_image_path))+"_"+str(os.path.basename(values[0]).replace('.nii','').replace('.IMA','').replace('.dcm',''))+"_"+str(f_Filter_Key.get())+"_concat.nii"

            print(file_name_from_firs_file)
            # Define the output file path
            output_file_path = os.path.join(f_save_image_path, file_name_from_firs_file)
            
            if os.path.exists(output_file_path):
                result = tk.messagebox.askyesno("[info]", "The file already exists. Do you want to proceed?")
                if result:
                    # print("save1")
                    # Save the NIfTI image to file
                    nib.save(nifti_img, output_file_path.replace("'","").replace(" ", "_"))
                    window_filter.lift()
                else:
                    # Save the DICOM file
                    file_name_from_firs_file = str(os.path.basename(f_image_path))+"_"+str(os.path.basename(values[0]).replace('.nii','').replace('.IMA','').replace('.dcm',''))+"_"+str(f_Filter_Key.get().replace("\n", ""))+"_concat_(2).nii"
        
                    # print("save2")
                    # Define the output file path
                    output_file_path = os.path.join(f_save_image_path, file_name_from_firs_file)
                    # Save the NIfTI image to file
                    nib.save(nifti_img, output_file_path.replace("'","").replace(" ", "_"))
                    window_filter.lift()
            else:
                nib.save(nifti_img, output_file_path.replace("'","").replace(" ", "_"))
                   
            output_file_path = output_file_path.replace("'","").replace(" ", "_")
            print(f"Saved concatenated image as NIfTI to {output_file_path}")
            tk.messagebox.showwarning("[info]", f"Saved concatenated image as NIfTI to {output_file_path}")
            window_filter.lift()

            log_file_path = 'log_file.xlsx'  # Path to your log Excel file
            folder_name = os.path.basename(f_image_path)   # Example folder name
            file_name = file_name_from_firs_file.replace("'","").replace(" ", "_")    # Example file name
            # Append a new log entry
            append_log_to_excel(log_file_path, folder_name, file_name)



        else:
            tk.messagebox.showwarning("[info]", "Select Images to Stack!!!")
            window_filter.lift()



    def f_save_dir():
        global f_save_image_path
        try:
            f_save_image_path = filedialog.askdirectory(title="Select a Directory")
            save_dir_path1.set("Save Directory : "+str(f_save_image_path))
            save_dir_path1.set(truncate_path(save_dir_path1.get()))


        except:
            pass
        window_filter.lift()

    def copy_move():
        global file_images
        global f_save_image_path
        global f_image_path



        try:
            # Setup progress bar
            progbar = ttk.Progressbar(window_filter, orient=tk.HORIZONTAL, length=800, mode='determinate')
            progbar.grid(column=1, row=6, columnspan=2, sticky="new")
            # Iterate over selected items
            for index, item in enumerate(f_lb.get_children()):
                # Update progress bar
                progbar['value'] = (index + 1) / len(f_lb.get_children()) * 100
                progbar.update_idletasks()
                window_filter.update()
                progbar.update()

                # Process item if selected
                values = list(f_lb.item(item, "values"))
                filename, checkbox_indicator = values[1], values[3]
                if checkbox_indicator == "[x]":
                    # full_path = os.path.join(f_image_path, filename)
                    shutil.copy(os.path.join(f_image_path, filename),   os.path.join(f_save_image_path, filename))
            progbar.destroy()


        except:
            tk.messagebox.showwarning("[info]", "Select a Directory")
            window_filter.lift()
            return





    def show_img_info():
        global f_image_path

        if f_image_path==None:
            tk.messagebox.showwarning("[info]", "Select a Directory")
            window_filter.lift()
            return

        selected_item = f_lb.selection()
        values = (f_lb.item(selected_item, "values"))
        selected_img =values[1]


        new_window = tk.Toplevel(root)
        new_window.title("[Image Header Info...]")

        # Set the geometry (size and position) of the new window if desired
        new_window.geometry("1000x800")  # Width x Height in pixels

        # Create a Text widget and a Scrollbar
        text_widget = tk.Text(new_window, wrap="word")
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(new_window, command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")

        if selected_item:  # Ensure there's a selection
            item = selected_item[1]  # Get the first (or only) selected item
            item_values = f_lb.item(item, 'values')  # Get the text of the item
            # select_img.set(str(os.path.basename(item_values[0]).replace("\\","/")))

            if item_values[1].endswith(".nii"):
                header = nib.load(os.path.join(f_image_path,selected_img)).header
                message = str(header)
            else:
                filepath = os.path.join(f_image_path,selected_img)
                ds = pydicom.dcmread(filepath)

                message = str(ds)
        else:
            pass

            message = "[info] Warning: please seclect an image for header information!!!"

        # Insert the message into the Text widget and disable editing
        text_widget.insert("1.0", message)
        text_widget.config(state="disabled")

    def toggle_select_all():
        # Toggle the text and state
        if ckbox_opt.get() == "Select All: [ ]":
            ckbox_opt.set("Select All: [x]")


            if len(f_lb.get_children())>=1:
                for item in f_lb.get_children():
                    values = list(f_lb.item(item, "values"))
                    values[3] = "[x]"
                    f_lb.item(item, values=values)

            # Add additional functionality here for selecting all options
        else:
            ckbox_opt.set("Select All: [ ]")
            # Add additional functionality here for deselecting all options
            if len(f_lb.get_children())>=1:
                for item in f_lb.get_children():
                    values = list(f_lb.item(item, "values"))
                    values[3] = "[ ]"
                    f_lb.item(item, values=values)


    def open_excel_file():
        # Ask the user to select an Excel file
        # filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        filepath = 'log_file.xlsx'
        if filepath:  # Check if a file was selected
            try:
                window_filter.update()
                # Attempt to open the file using the default application
                os.system('start "excel" '+filepath)
                # subprocess.run(['open', filepath] if sys.platform == 'darwin' else ['start', filepath], check=True, shell=True)
            except Exception as e:
                print(f"Error opening file: {e}")
                return



    def delete_selected_records():
        for item in f_lb.get_children():  # Iterate over all items in the Treeview
            values = f_lb.item(item, "values")
            if values[3] == "[x]":  # Check if the third value is "[x]"
                # Construct the full file path
                file_path = os.path.join(f_image_path, values[1])  # Assuming values[0] contains the file name
                try:
                    # Delete the file from the filesystem
                    os.remove(file_path)
                    # Delete the item from the Treeview
                    f_lb.delete(item)
                    print(f"Deleted file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e.strerror}")
        # Update the count after deletion
        f_count_files.set('Total Number of Files: ' + str(len(f_lb.get_children())))
    
    def truncate_path(path, max_length=120):
        """Truncate a file path to fit within a maximum length, preserving the start and end of the path."""
        if len(path) <= max_length:
            return path
        else:
            # part_length = (max_length - 3) // 2  # Subtract 3 for ellipsis
            part_length = (max_length) // 2  # Subtract none for ellipsis
            return f'{path[:part_length]}...{path[-part_length:]}'
   
    
   # Function to handle mouse scroll
    def mouse_scroll(event):
        if event.delta:
            f_lb.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            if event.num == 5:
                move = 1
            else:
                move = -1
            f_lb.yview_scroll(move, "units")
            
    def chek_processed(path):
        def read_folder_names_from_excel(excel_path):
            df = pd.read_excel(excel_path, engine='openpyxl')
            return df['Folder Name'].unique()
        
        folder_names = read_folder_names_from_excel('./log_file.xlsx')
        if any(pt in f_image_path for pt in folder_names):
            return True
        

    def open_next_directory():
        global f_image_path
        if not f_image_path:
            tk.messagebox.showinfo("Error", "No directory is currently selected.")
            window_filter.lift()
            return
    
        parent_directory = os.path.dirname(f_image_path)
        directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
        # directories.sort()  # Sort the directories to ensure a consistent order
        directories = sorted(directories, key=lambda x: os.path.basename(x).lower())
    
        current_directory_name = os.path.basename(f_image_path)
        if current_directory_name in directories:
            current_index = directories.index(current_directory_name)
            next_index = current_index + 1 if current_index + 1 < len(directories) else 0  # Loop back to the first directory
              
            next_directory = os.path.join(parent_directory, directories[next_index])
            f_image_path = next_directory
            open_dir_path1.set("Open Directory: " + f_image_path)  # Update the displayed path
            # You may want to call the function to reload the file list here, similar to f_open_dir() but for the new directory
            f_open_dir_next_prev(f_image_path)  # Assuming this function reloads the file list for the new directory
        else:
            tk.messagebox.showinfo("Error", "The current directory is not found in its parent directory.")
            window_filter.lift()
        
    def open_previous_directory():
        global f_image_path
        if not f_image_path:
            tk.messagebox.showinfo("Error", "No directory is currently selected.")
            window_filter.lift()
            return
    
        parent_directory = os.path.dirname(f_image_path)
        directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
        # directories.sort()  # Ensure the directories are in a consistent order
        directories = sorted(directories, key=lambda x: os.path.basename(x).lower())
        current_directory_name = os.path.basename(f_image_path)
        if current_directory_name in directories:
            current_index = directories.index(current_directory_name)
            previous_index = current_index - 1 if current_index - 1 >= 0 else len(directories) - 1  # Loop to the last directory if at the first
            previous_directory = os.path.join(parent_directory, directories[previous_index])
            f_image_path = previous_directory
            open_dir_path1.set("Open Directory: " + f_image_path)  # Update the displayed path
            # Here you might want to call the function to reload the file list for the new directory
            f_open_dir_next_prev(f_image_path) # Assuming this function reloads the file list for the new directory
        else:
            tk.messagebox.showinfo("Error", "The current directory is not found in its parent directory.")
            window_filter.lift()
            
    open_dir_info= tk.Label(window_filter,textvariable= open_dir_path1, justify=tk.LEFT)
    open_dir_info.config(font=('courier',12))
    open_dir_info.grid(column=1,row=6,columnspan=2,sticky="nw")




    save_dir_info= tk.Label(window_filter,textvariable= save_dir_path1, justify=tk.LEFT)
    save_dir_info.config(font=('courier',12))
    save_dir_info.grid(column=1,row=6,columnspan=2,sticky="ws")
    


    btn_open = tk.Button(window_filter, text='Open folder',command=f_open_dir, height = 1,  width = 20 )
    btn_open.config(font=('courier',12))
    btn_open.grid(column=1,row=7, sticky="w",padx= (50,50))

    btn_save = tk.Button(window_filter, text='Save folder',command=f_save_dir, height = 1,  width = 20 )
    btn_save.config(font=('courier',12))
    btn_save.grid(column=1,row=8, sticky="w",padx= (50,50))

    btn_log = tk.Button(window_filter, text='Log File',command=open_excel_file, height = 1,  width = 20 )
    btn_log.config(font=('courier',12))
    btn_log.grid(column=0,row=8, sticky="w",padx= (50,50))


    btn_search = tk.Button(window_filter, text='Filter',command=f_search_dir, height = 1,  width = 20 )
    btn_search.config(font=('courier',12))
    btn_search.grid(column=1,row=7,columnspan=1, sticky="e", pady=(5, 5),padx= (50,50))

    btn_info = tk.Button(window_filter, text='Image Info',command=show_img_info, height = 1,  width = 20 )
    btn_info.config(font=('courier',12))
    btn_info.grid(column=1,row=7,columnspan=1, sticky="", pady=(5, 5),padx= (50,50))


    btn_close = tk.Button(window_filter, text='Close',command= close_window, height = 1,  width = 20) #
    btn_close.config(font=('courier',12))
    btn_close.grid(column=2,row=8, sticky="e",padx= (50,50))

    btn_copy = tk.Button(window_filter, text='Copy Images',command= copy_move, height = 1,  width = 20) #
    btn_copy.config(font=('courier',12))
    btn_copy.grid(column=1,row=8, sticky="",padx= (50,50))

    btn_comp = tk.Button(window_filter, text='Compress Images',command= comp_move, height = 1,  width = 20) #
    btn_comp.config(font=('courier',12))
    btn_comp.grid(column=1,row=8, sticky="e",padx= (50,50))

    btn_del = tk.Button(window_filter, text='Delete Record',command= delete_selected_records, height = 1,  width = 20) #
    btn_del.config(font=('courier',12))
    btn_del.grid(column=2,row=7, sticky="e",padx= (50,50))
    # btn_del.grid(column=2,row=8, sticky="")

    btn_nxt = tk.Button(window_filter, text='Next Dir',command= open_next_directory, height = 1,  width = 20) #
    btn_nxt.config(font=('courier',12))
    btn_nxt.grid(column=2,row=8, sticky="",padx= (50,50))
    
    btn_prev = tk.Button(window_filter, text='Previous DIr',command= open_previous_directory, height = 1,  width = 20) #
    btn_prev.config(font=('courier',12))
    btn_prev.grid(column=2,row=8, sticky="w",padx= (50,50))
    

    # Name the custom style. It inherits from the default Treeview style.
    style_name = "Custom.Treeview"

    # Configure the custom style
    style.configure(style_name, rowheight=60,  font=('courier',16))  # Set the desired rowheight

    # Create the Treeview widget with the custom style
    f_lb = ttk.Treeview(window_filter, columns=("Image", "idx" ,"Name","Description", "Checkbox"), style=style_name)
    f_lb.heading('#0', text='Image', anchor=tk.CENTER)
    f_lb.heading('#1', text='idx', anchor=tk.CENTER)
    f_lb.heading('#2', text='File Name', anchor=tk.CENTER)
    f_lb.heading('#3', text='Description', anchor=tk.CENTER)
    f_lb.heading("#4", text="Select", anchor=tk.CENTER)


    f_lb.column('#0', stretch=tk.NO, width=400, anchor=tk.CENTER)
    f_lb.column('#1', stretch=tk.NO, width=100, anchor=tk.CENTER)
    f_lb.column('#2', stretch=tk.NO, width=600, anchor=tk.W)
    f_lb.column('#3', stretch=tk.NO, width=400, anchor=tk.W)
    f_lb.column('#4', stretch=tk.NO, width=200, anchor=tk.CENTER)  # Adjust column width as needed

    f_lb.grid(column=1, row=2, columnspan=2, rowspan=4, sticky="nsew")
    f_lb.bind("<<TreeviewSelect>>", f_callback)
    # Bind a click event to toggle the checkbox state
    f_lb.bind('<Button-1>', lambda event: toggle_checkbox(f_lb.identify_row(event.y)))

    # Create a Scrollbar and associate it with the Treeview
    scrollbar = ttk.Scrollbar(window_filter, orient="vertical", command=f_lb.yview)
    f_lb.configure(yscrollcommand=scrollbar.set)

    # Position the Treeview and the Scrollbar in the window
    f_lb.grid(column=1, row=2, columnspan=2, rowspan=4, sticky="nsew")
    scrollbar.grid(column=3, row=2, rowspan=4, sticky='ns')
    
    
    
    # Binding mouse scroll events
    window_filter.bind("<MouseWheel>", mouse_scroll)  # For Windows and MacOS
    # window_filter.bind("<Button-4>", mouse_scroll)  # For Linux, scrolling up
    # window_filter.bind("<Button-5>", mouse_scroll)  # For Linux, scrolling down

    f_textBox = tk.Entry(window_filter, width=30, textvariable= f_Filter_Key,font=('courier',12))
    f_textBox.configure(justify=tk.CENTER)
    f_textBox.grid(column=2,row=7,columnspan=1, sticky="", pady=(10, 10),padx= (25,25))


    f_key_files =tk.StringVar()
    f_key_files.set("Enter Search Keyword : ")

    f_keyword_label = tk.Label(window_filter,textvariable=f_key_files)
    f_keyword_label.config(font=('courier',12),justify=tk.RIGHT)
    f_keyword_label.grid(column=2,row=7,columnspan=1, sticky="w",padx= (50,50))

    f_count_label = tk.Label(window_filter,textvariable=f_count_files)
    f_count_label.config(font=('courier',12))
    f_count_label.grid(column=2,row=6,columnspan=1,sticky="ne")



    # Create a frame for background
    background_frame = tk.Frame(window_filter, bg='lightgray', width=200, height=50)
    background_frame.grid(column=2, row=0, columnspan=1, sticky="es")
    background_frame.grid_propagate(False)  # Prevents the frame from resizing to fit the label

    # StringVar to keep track of the checkbox option
    ckbox_opt = tk.StringVar()
    ckbox_opt.set('Select All: [ ]')

    # Create a label within the background frame
    ckbox_label = tk.Label(background_frame, textvariable=ckbox_opt, bg='lightgray')
    ckbox_label.config(font=('courier', 12))
    ckbox_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the label in the frame

    # Bind the click event to the label
    ckbox_label.bind("<Button-1>", lambda e: toggle_select_all())







    window_filter.columnconfigure(0, weight=1)
    window_filter.columnconfigure(1, weight=1)
    window_filter.columnconfigure(2, weight=1)
    window_filter.columnconfigure(3, weight=1)
    window_filter.columnconfigure(4, weight=1)


    window_filter.rowconfigure(0, weight=1)
    window_filter.rowconfigure(1, weight=1)
    window_filter.rowconfigure(2, weight=1)
    window_filter.rowconfigure(3, weight=1)
    window_filter.rowconfigure(4, weight=1)
    window_filter.rowconfigure(5, weight=1)
    window_filter.rowconfigure(6, weight=1)
    window_filter.rowconfigure(7, weight=1)
    window_filter.rowconfigure(8, weight=1)



def show_img_info():

    selected_item = lb.selection()

    # print(selected_item)


    new_window = tk.Toplevel(root)
    new_window.title("[Image Header Info...]")

    # Set the geometry (size and position) of the new window if desired
    new_window.geometry("1000x800")  # Width x Height in pixels

    # Create a Text widget and a Scrollbar
    text_widget = tk.Text(new_window, wrap="word")
    text_widget.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(new_window, command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")

    if selected_item:  # Ensure there's a selection
        item = selected_item[0]  # Get the first (or only) selected item
        item_values = lb.item(item, 'values')  # Get the text of the item
        # select_img.set(str(os.path.basename(item_values[0]).replace("\\","/")))

        if item_values[0].endswith(".nii"):
            header = nib.load(os.path.join(image_path,str(select_img.get()))).header
            message = str(header)
        else:
            filepath = os.path.join(image_path, str(select_img.get()))
            ds = pydicom.dcmread(filepath)

            message = str(ds)
    else:
        pass

        message = "[info] Warning: please seclect an image for header information!!!"

    # Insert the message into the Text widget and disable editing
    text_widget.insert("1.0", message)
    text_widget.config(state="disabled")


global img_W,img_H


def select_image():
    image = lb.get(lb.curselection())
    select_img.set((image))
    gc.collect()


def select_nii():
    global img_W,img_H

    try:
        # if os.path.exists('./result'):
        #     os.remove('./result')
        os.mkdir('./result')
    except:
        pass

    header = nib.load(os.path.join(image_path,str(select_img.get()))).header
    pixel_area= header.get_zooms()[0]*header.get_zooms()[0]

    img_W = (header["dim"][1])
    img_H = (header["dim"][2])
    slices = (header["dim"][3])
    img_details.set("[info]:\n"+
                    "pixel_area:"+'{0:.3f}'.format(pixel_area)+"\n"+
                    "Width:"+str(img_W)+"\n"+
                    "Height:"+str(img_H)+"\n"+
                    "Slices:"+str(slices)+"\n")
    img_unit_top.set("\n"+
                 "mm\u00b2"+"\n"+
                 "pixcel"+"\n"+
                 "pixcel"+"\n")


    image_array = nib.load(os.path.join(image_path,str(select_img.get()))).get_fdata()[:, :, 0]
    
    
    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

    # Avoid division by zero by ensuring the maximum value is greater than zero
    max_val = np.max(image_array)
    if max_val > 0:
        # Normalize the image data to 0-255
        image_array = (image_array / max_val * 255).astype(np.uint8)
    else:
        # Handle the case where the image is completely black or max_val is 0
        image_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale if not already
    
    
    
    
    
    # image = Image.fromarray(image)
    global image_on_canvas_ref  # Use this to keep a reference to the image to prevent garbage collection
    if img_W<=800 and img_H<=600:
        New_W = int(img_W*(800/img_W))
        New_H = int(img_H*(600/img_H))
    else:
        New_W = int(img_W*(img_W/800))
        New_H = int(img_H*(img_H/600))

    image = image.resize((New_W, New_H))
    tk_photo = ImageTk.PhotoImage(image, master=root)  # Ensure master is correctly set

    # Assuming `image_on_canvas` is the ID of the initially created image on the canvas
    canvas.itemconfig(image_on_canvas, image=tk_photo)
    image_on_canvas_ref = tk_photo

    if slices>=2:
        trackbar.configure(from_=1,to=slices)
        trackbar.config(state="normal")
    else:
        trackbar.configure(from_=1, to=2)
        trackbar.config(state="normal")
        
    trackbar.set(1)
    save_folder_name = pathlib.Path(str(select_img.get())).stem
    if os.path.exists('./result/'+save_folder_name):
        update_canvas_with_image('./result/'+save_folder_name+'/1.png')
        
    gc.collect()


def select_dicom():
    global img_W,img_H
    try:
        # if os.path.exists('./result'):
        #     os.remove('./result')
        os.mkdir('./result')
    except:
        pass

    # Assuming select_img.get() returns the filename of the selected DICOM file
    filepath = os.path.join(image_path, str(select_img.get()))

    # Load the DICOM file
    ds = pydicom.dcmread(filepath)

    # Pixel spacing (assumes 2D image). For volumetric data, you might have Slice Thickness (0018,0050) and/or Spacing Between Slices (0018,0088)
    pixel_spacing = ds.PixelSpacing if 'PixelSpacing' in ds else [1, 1]  # Default to 1 if not specified
    pixel_area = pixel_spacing[0] * pixel_spacing[1]

    # Dimensions - The Width and Height of the image
    img_W, img_H = ds.Rows, ds.Columns

    # For volumetric data, you might use the 'NumberOfFrames' attribute if present
    slices = ds.NumberOfFrames if 'NumberOfFrames' in ds else 1  # Default to 1 if not specified or 2D image

    img_details.set("[info]:\n" +
                    "pixel_area:" + '{0:.3f}'.format(pixel_area) + "\n" +
                    "Width:" + str(img_W) + "\n" +
                    "Height:" + str(img_H) + "\n" +
                    "Slices:" + str(slices) + "\n")
    img_unit_top.set("\n" +
                     "mm\u00b2" + "\n" +
                     "pixel" + "\n" +
                     "pixel" + "\n")



    image_array = ds.pixel_array

    if len(image_array.shape)>=3:
        image_array = image_array[:, :, 0]


    image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

    # Avoid division by zero by ensuring the maximum value is greater than zero
    max_val = np.max(image_array)
    if max_val > 0:
        # Normalize the image data to 0-255
        image_array = (image_array / max_val * 255).astype(np.uint8)
    else:
        # Handle the case where the image is completely black or max_val is 0
        image_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)
    if image.mode != 'L':
        image = image.convert('L')  # Convert to grayscale if not already

    # image = Image.fromarray(image)
    global image_on_canvas_ref  # Use this to keep a reference to the image to prevent garbage collection
    if img_W<=800 and img_H<=600:
        New_W = int(img_W*(800/img_W))
        New_H = int(img_H*(600/img_H))
    else:
        New_W = int(img_W*(img_W/800))
        New_H = int(img_H*(img_H/600))

    image = image.resize((New_W, New_H))
    tk_photo = ImageTk.PhotoImage(image, master=root)  # Ensure master is correctly set

    # Assuming `image_on_canvas` is the ID of the initially created image on the canvas
    canvas.itemconfig(image_on_canvas, image=tk_photo)
    image_on_canvas_ref = tk_photo



    if slices>=2:
        trackbar.configure(from_=1,to=slices)
        trackbar.config(state="normal")
    else:
        # print(True)
        trackbar.configure(from_=1, to=2)
        trackbar.config(state="normal")
        
    save_folder_name = pathlib.Path(str(select_img.get())).stem
    if os.path.exists('./result/'+save_folder_name):
        update_canvas_with_image('./result/'+save_folder_name+'/1.png')    
        
    trackbar.set(1)
    gc.collect()


def close_window():
    root.destroy()
    sys.exit()

def deactivate_button():
    btn_close.config(state="disable")
    btn_go.config(state="disable")
    btn_open.config(state="disable")
    btn_calculate.config(state="disable")
    btn_info.config(state="disable")
    btn_filter.config(state="disable")
    btn_Save.config(state="disable")
    btn_sorter.config(state="disable")


def activate_button():
    btn_close.config(state="normal")
    btn_go.config(state="normal")
    btn_open.config(state="normal")
    btn_calculate.config(state="normal")
    btn_info.config(state="normal")
    btn_filter.config(state="normal")
    btn_Save.config(state="normal")
    btn_sorter.config(state="normal")

def cal_ad(AoMax,AoMin,pp):
    A =(AoMax-AoMin)
    B = AoMin*pp
    return A/B



# def pred_init():
#     global rotation_angle
#     if int(mode_var.get()) ==1:
#         try:
#             predict(select_img.get(),True)
#         except:
#             try:
#                 rotation_angle=rotation_angle+90
#                 predict(select_img.get(),True)
#             except:
#                 try:
#                     rotation_angle=rotation_angle+90
#                     predict(select_img.get(),True)
#                 except:
#                     try:
#                         rotation_angle=rotation_angle+90
#                         predict(select_img.get(),True)
#                     except:
#                         print(select_img.get(),"Failed!!")
#                         pass
                
#         trackbar.config(state="normal")
#     elif int(mode_var.get()) ==2:
        
#         lb.state(('disabled',))
#         progbar = ttk.Progressbar(root,  orient=tk.HORIZONTAL, length=800, mode='determinate')
#         progbar.grid(column=1,row=5,rowspan=2,sticky="w")
#         i = 0
#         for item in lb.get_children():
#             progbar['value'] = (int(i+1)/int(len(lb.get_children())))*100
#             progbar.update_idletasks()
#             progbar.update()
#             root.update()

#             try:
#                 values = list(lb.item(item, "values"))[0]
#                 predict(values, False)
#             except:
#                 try:
#                     rotation_angle=rotation_angle+90
#                     values = list(lb.item(item, "values"))[0]
#                     predict(values, False)
#                 except:
#                     try:
#                         rotation_angle=rotation_angle+90
#                         values = list(lb.item(item, "values"))[0]
#                         predict(values, False)
#                     except:
#                         try:
#                             rotation_angle=rotation_angle+90
#                             values = list(lb.item(item, "values"))[0]
#                             predict(values, False)
#                         except:
#                             print(select_img.get(),"Failed!!")
#                             continue
            
#             i = i+1
        
#         progbar['value'] = 100
#         progbar.update_idletasks()
#         progbar.update()
#         root.update()
#         trackbar.set(1)
        
#         activate_button()
#         progbar.destroy()
#         lb.state(('!disabled',))

#         trackbar.config(state="normal")
        
def pred_init():
    global rotation_angle
    if int(mode_var.get()) ==1:
        predict(select_img.get(),True)
        root.update()
        trackbar.config(state="normal")
    elif int(mode_var.get()) ==2:
        
        lb.state(('disabled',))
        progbar = ttk.Progressbar(root,  orient=tk.HORIZONTAL, length=800, mode='determinate')
        progbar.grid(column=1,row=5,rowspan=2,sticky="w")
        i = 0
        for item in lb.get_children():
            progbar['value'] = (int(i+1)/int(len(lb.get_children())))*100
            progbar.update_idletasks()
            progbar.update()
            root.update()
            values = list(lb.item(item, "values"))[1]
            predict(values, False)
                        
            i = i+1
            root.update()
        
        progbar['value'] = 100
        progbar.update_idletasks()
        progbar.update()
        root.update()
        trackbar.set(1)
        
        activate_button()
        progbar.destroy()
        lb.state(('!disabled',))

        trackbar.config(state="normal")    
    
    
def predict(input_image_pred, single_mode = True):
    global rotation_angle
    deactivate_button()
    
    if single_mode:
        lb.state(('disabled',))
        progbar = ttk.Progressbar(root,  orient=tk.HORIZONTAL, length=800, mode='determinate')
        progbar.grid(column=1,row=5,rowspan=2,sticky="w")
    
    psudo_lbl = []
    save_folder_name = pathlib.Path(str(input_image_pred)).stem
    
    
    
    try:
        if os.path.exists('./result/'+save_folder_name):
            os.remove('./result/'+save_folder_name)
        os.mkdir('./result/'+save_folder_name)
        os.mkdir('./psudo_labels')
        
        rotation_angle=0
    except:
        pass
    
    
    try:
        if os.path.exists('./psudo_labels'):
            os.remove('./psudo_labels')
        os.mkdir('./psudo_labels')
    
    except:
        pass

    # print(str(select_img.get()))
    if str(input_image_pred).endswith("nii"):
        test_img = nib.load(os.path.join(image_path,str(input_image_pred))).get_fdata()
    else:
        dicom_data = pydicom.dcmread(os.path.join(image_path,str(input_image_pred)))
        test_img = dicom_data.pixel_array

    if rotation_angle==90:
        test_img = np.rot90(test_img)
    elif rotation_angle==180:
        test_img = np.rot90(test_img)
        test_img = np.rot90(test_img)
    elif rotation_angle==270:
        test_img = np.rot90(test_img)
        test_img = np.rot90(test_img)
        test_img = np.rot90(test_img)
    else:
        pass



    if len(test_img.shape)>=3:

        for i in tqdm(range(test_img.shape[2])):
            if single_mode:
                progbar['value'] = (int(i+1)/int(test_img.shape[2]))*100
                progbar.update_idletasks()
                progbar.update()
                root.update()

            # print((int((i+1)/int(test_img.shape[2]))*1000))


            fr_img = test_img[:,:,i]
            fr_img = np.expand_dims(fr_img, axis=2)


            if fr_img.shape[0]<256 or fr_img.shape[1]<256:
                 fr_img = pad(fr_img, 256, 256)
            elif fr_img.shape[0]>256 or fr_img.shape[1]>256:
                fr_img = cv2.resize(fr_img, (256, 256))


            input_img = np.expand_dims(fr_img, axis=0)

            output_img = model.predict(input_img)

            output_img = np.squeeze(output_img, axis=0)

            mean = output_img.mean()
            output_img[output_img >= mean] = 1
            output_img[output_img < mean] = 0
            try:
                output_img = assign_lbl(output_img)
                path = './result/'+save_folder_name+"/"+str(i+1)+'.png'
                save_img(fr_img,output_img,path,i)
                psudo_lbl.append(output_img)
                # print(output_img.shape)
            except:
                 path = './result/'+save_folder_name+"/"+str(i+1)+'.png'
                 save_img(fr_img,fr_img,path,i)  
                 psudo_lbl.append(np.zeros((256, 256)))
                 # print(output_img.shape)


            

            

        trackbar.set(1)
    else:
        if single_mode:
            progbar['value'] = 50
            progbar.update_idletasks()
            progbar.update()
            root.update()



        fr_img = test_img
        fr_img = np.expand_dims(fr_img, axis=2)


        if fr_img.shape[0]<256 or fr_img.shape[1]<256:
            fr_img = pad(fr_img, 256, 256)
        elif fr_img.shape[0]>256 or fr_img.shape[1]>256:
            fr_img = cv2.resize(fr_img, (256, 256))

        input_img = np.expand_dims(fr_img, axis=0)

        output_img = model.predict(input_img)

        output_img = np.squeeze(output_img, axis=0)
        if fr_img.shape[0]>256 or fr_img.shape[1]>256:
            output_img = np.reshape(output_img, newshape=(fr_img.shape[0], fr_img.shape[1]))

        mean = output_img.mean()
        output_img[output_img >= mean] = 1
        output_img[output_img < mean] = 0

        # output_img = assign_lbl(output_img)
        # path = './result/'+save_folder_name+"/"+str(1)+'.png'

        try:
            output_img = assign_lbl(output_img)
            path = './result/'+save_folder_name+"/"+str(i+1)+'.png'
            save_img(fr_img,output_img,path,0)
            psudo_lbl.append(output_img)
        except:
             path = './result/'+save_folder_name+"/"+str(i+1)+'.png'
             save_img(fr_img,fr_img,path,0)    
             psudo_lbl.append(np.zeros((256, 256)))
             
        # save_img(fr_img,output_img,path,0)
        
        
        if single_mode:
            progbar['value'] = 100
            progbar.update_idletasks()
            progbar.update()
            root.update()
            trackbar.set(1)

    
    psudo_lbl = np.array(psudo_lbl)

    psudo_lbl = np.transpose(psudo_lbl, (1, 2, 0))
    # np.save("./psudo_labels/"+save_folder_name+".npy", psudo_lbl)
    

    filename = f"./psudo_labels/{save_folder_name}.nii"
    
    # Create a Nifti1Image
    nifti_img = nib.Nifti1Image(psudo_lbl, affine=np.eye(4))
    
    # Save the NIfTI image
    nib.save(nifti_img, filename)
    

    # try:
    #     if len(test_img.shape)>=3:

    #         for i in range(test_img.shape[2]):
    #             progbar['value'] = (int(i+1)/int(test_img.shape[2]))*100
    #             progbar.update_idletasks()
    #             progbar.update()
    #             root.update()

    #             # print((int((i+1)/int(test_img.shape[2]))*1000))


    #             fr_img = test_img[:,:,i]
    #             fr_img = np.expand_dims(fr_img, axis=2)


    #             if fr_img.shape[0]<256 or fr_img.shape[1]<256:
    #                 fr_img = pad(fr_img, 256, 256)


    #             input_img = np.expand_dims(fr_img, axis=0)

    #             output_img = model.predict(input_img)

    #             output_img = np.squeeze(output_img, axis=0)

    #             mean = output_img.mean()
    #             output_img[output_img >= mean] = 1
    #             output_img[output_img < mean] = 0

    #             output_img = assign_lbl(output_img)
    #             path = './result/'+str(i+1)+'.png'



    #             save_img(fr_img,output_img,path,i)

    #             psudo_lbl.append(output_img)

    #         trackbar.set(1)
    #     else:
    #         progbar['value'] = 50
    #         progbar.update_idletasks()
    #         progbar.update()
    #         root.update()



    #         fr_img = test_img
    #         fr_img = np.expand_dims(fr_img, axis=2)


    #         if fr_img.shape[0]<256 or fr_img.shape[1]<256:
    #             fr_img = pad(fr_img, 256, 256)
    #         elif fr_img.shape[0]>256 or fr_img.shape[1]>256:
    #             fr_img = cv2.resize(fr_img, (256, 256))

    #         input_img = np.expand_dims(fr_img, axis=0)

    #         output_img = model.predict(input_img)

    #         output_img = np.squeeze(output_img, axis=0)
    #         if fr_img.shape[0]>256 or fr_img.shape[1]>256:
    #             output_img = np.reshape(output_img, newshape=(fr_img.shape[0], fr_img.shape[1]))

    #         mean = output_img.mean()
    #         output_img[output_img >= mean] = 1
    #         output_img[output_img < mean] = 0

    #         output_img = assign_lbl(output_img)
    #         path = './result/'+str(1)+'.png'


    #         save_img(fr_img,output_img,path,0)
    #         psudo_lbl.append(output_img)
    #         progbar['value'] = 100
    #         progbar.update_idletasks()
    #         progbar.update()
    #         root.update()
    #         trackbar.set(1)


    #     psudo_lbl = np.array(psudo_lbl)
    #     np.save("./result/psudo_segmentation.npy", psudo_lbl)

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     tk.messagebox.showwarning("Warning", "This is not suitable to measure AD!!")
    #     lb.state(('!disabled',))
    #     activate_button()

    #     progbar.destroy()
    #     tf.keras.backend.clear_session()
    #     gc.collect()
    #     pass


    
    if single_mode:
        activate_button()
        progbar.destroy()
        lb.state(('!disabled',))

    trackbar.config(state="normal")
    pixel_area = float(img_details.get().split("\n")[1].replace("pixel_area:",""))
    AAo_arr = np.array(AAo)
    DAo_arr = np.array(DAo)
    img_area.set("AAo:"+'{0:.3f}'.format(AAo_arr[0]*pixel_area)+"\n"+
                 "DAo:"+'{0:.3f}'.format(DAo_arr[0]*pixel_area)+"\n"+
                 "AAo_Max:"+'{0:.3f}'.format(AAo_arr.max()*pixel_area)+"\n"+
                 "DAo_Max:"+'{0:.3f}'.format(DAo_arr.max()*pixel_area)+"\n"+
                 "AAo_Min:"+'{0:.3f}'.format(AAo_arr.min()*pixel_area)+"\n"+
                 "DAo_Min:"+'{0:.3f}'.format(DAo_arr.min()*pixel_area)+"\n")
    try:
        update_canvas_with_image('./result/'+save_folder_name+'/1.png')
    except:
        update_canvas_with_image('logo.png')

    np.save('./result/'+save_folder_name+'/AAo_arr.npy',AAo_arr)
    np.save('./result/'+save_folder_name+'/DAo_arr.npy',DAo_arr)


    
    tf.keras.backend.clear_session()
    gc.collect()
    if rotation_angle>=90:
        rotation_angle=0

def update_canvas_with_image(image_path):
    global rotation_angle
    """Update the canvas with a new image given by image_path."""
    global image_on_canvas_ref  # Use this to keep a reference to the image to prevent garbage collection
    pil_image = Image.open(image_path)
    tk_photo = ImageTk.PhotoImage(pil_image, master=root)  # Ensure master is correctly set

    # Assuming `image_on_canvas` is the ID of the initially created image on the canvas
    canvas.itemconfig(image_on_canvas, image=tk_photo)
    image_on_canvas_ref = tk_photo

##    update_image()

def update_canvas(value):
    global rotation_angle
    # Assuming AAo and DAo are previously defined lists or arrays
    try:
        try:   
            AAo_arr = np.array(AAo)
            DAo_arr = np.array(DAo)
            pixel_area = float(img_details.get().split("\n")[1].replace("pixel_area:", ""))
        
            img_area.set(
                "AAo:" + '{0:.3f}'.format(AAo_arr[int(value)-1] * pixel_area) + "\n" +
                "DAo:" + '{0:.3f}'.format(DAo_arr[int(value)-1] * pixel_area) + "\n" +
                "AAo_Max:" + '{0:.3f}'.format(AAo_arr.max() * pixel_area) + "\n" +
                "DAo_Max:" + '{0:.3f}'.format(DAo_arr.max() * pixel_area) + "\n" +
                "AAo_Min:" + '{0:.3f}'.format(AAo_arr.min() * pixel_area) + "\n" +
                "DAo_Min:" + '{0:.3f}'.format(DAo_arr.min() * pixel_area) + "\n"
            )
        except:
            save_folder_name = pathlib.Path(str(select_img.get())).stem
            AAo_arr = np.load('./result/'+save_folder_name+'/AAo_arr.npy')
            DAo_arr =np.load('./result/'+save_folder_name+'/DAo_arr.npy')
              
            pixel_area = float(img_details.get().split("\n")[1].replace("pixel_area:", ""))
        
            img_area.set(
                "AAo:" + '{0:.3f}'.format(AAo_arr[int(value)-1] * pixel_area) + "\n" +
                "DAo:" + '{0:.3f}'.format(DAo_arr[int(value)-1] * pixel_area) + "\n" +
                "AAo_Max:" + '{0:.3f}'.format(AAo_arr.max() * pixel_area) + "\n" +
                "DAo_Max:" + '{0:.3f}'.format(DAo_arr.max() * pixel_area) + "\n" +
                "AAo_Min:" + '{0:.3f}'.format(AAo_arr.min() * pixel_area) + "\n" +
                "DAo_Min:" + '{0:.3f}'.format(DAo_arr.min() * pixel_area) + "\n"
            )
    
        # Correctly getting the path to the image file
        save_folder_name = pathlib.Path(str(select_img.get())).stem
        resfolder = sorted(glob.glob(os.path.join(os.getcwd(), 'result',save_folder_name, '*.png')), key=os.path.getmtime)
    
        image_path_n = resfolder[int(value)-1]
    
        # Load the image using PIL and convert it to PhotoImage
        pil_image = Image.open(image_path_n)
        tk_photo = ImageTk.PhotoImage(pil_image)
    
        # Update the canvas with the new image
        global image_on_canvas_ref  # This holds the PhotoImage to prevent garbage collection
        canvas.itemconfig(image_on_canvas, image=tk_photo)
    
        # Update the global reference to point to the new PhotoImage
        image_on_canvas_ref = tk_photo
    
        update_image()
    except:
        pixel_area = float(img_details.get().split("\n")[1].replace("pixel_area:", ""))
        
        img_area.set(
                "AAo:NA \n" +
                "DAo:NA\n" +
                "AAo_Max:NA\n" +
                "DAo_Max:NA\n" +
                "AAo_Min:NA\n" +
                "DAo_Min:NA\n"
            )
    
        # Correctly getting the path to the image file
        save_folder_name = pathlib.Path(str(select_img.get())).stem
        resfolder = sorted(glob.glob(os.path.join(os.getcwd(), 'result',save_folder_name, '*.png')), key=os.path.getmtime)
        try:
            image_path_n = resfolder[int(value)-1]
        
            # Load the image using PIL and convert it to PhotoImage
            pil_image = Image.open(image_path_n)
            tk_photo = ImageTk.PhotoImage(pil_image)
        
            # Update the canvas with the new image
            # global image_on_canvas_ref  # This holds the PhotoImage to prevent garbage collection
            canvas.itemconfig(image_on_canvas, image=tk_photo)
        
            # Update the global reference to point to the new PhotoImage
            image_on_canvas_ref = tk_photo
        except:
            pass
    
        update_image()
        
###############################################################################
zoom_level = 1
rotation_angle = 0

original_image = Image.open("logo.png")  # Assuming "logo.png" is your base image


# def adjust_contrast(image_array, lower_percentile=2, upper_percentile=85):
#     """
#     Adjust the contrast of an image based on the specified lower and upper percentiles.
#     This reduces brightness by spreading the intensity range.
#     """
#     lower_bound = np.percentile(image_array, lower_percentile)
#     upper_bound = np.percentile(image_array, upper_percentile)
    
#     # Clip the image array to the specified bounds and scale to 0-255
#     contrast_adjusted = np.clip(image_array, lower_bound, upper_bound)
#     contrast_adjusted = ((contrast_adjusted - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    
#     return contrast_adjusted


def update_image():
    global zoom_level, rotation_angle, canvas
    
    # Check the state of the button
    if btn_go['state'] != 'disabled':
        # If the button is enabled, load the image based on the trackbar value
        save_folder_name = pathlib.Path(str(select_img.get())).stem
        image_path = os.path.join(os.getcwd(), 'result',save_folder_name, str(int(trackbar.get())) + '.png')
        if os.path.exists(image_path):
            original_image = Image.open(image_path)



        else:
            # If the image doesn't exist, use the default one from path
            if str(select_img.get()).endswith("nii"):
                image_array = nib.load(os.path.join(image_path_unique,str(select_img.get()))).get_fdata()[:, :, 0]
                
                image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))
                 # Avoid division by zero by ensuring the maximum value is greater than zero
                max_val = np.max(image_array)
                if max_val > 0:
                    # Normalize the image data to 0-255
                    image_array = (image_array / max_val * 255).astype(np.uint8)
                    # image_array = adjust_contrast(image_array)
                else:
                    # Handle the case where the image is completely black or max_val is 0
                    image_array = np.zeros(image_array.shape, dtype=np.uint8)
                
                # Convert the numpy array to a PIL Image
                image = Image.fromarray(image_array)
                if image.mode != 'L':
                    image = image.convert('L')  # Convert to grayscale if not already
                # image = Image.fromarray(image_array)

            else:
                dicom_data = pydicom.dcmread(os.path.join(image_path_unique,str(select_img.get())))
                image_array = dicom_data.pixel_array

                if len(image_array.shape)>=3:
                    image_array = image_array[:, :, 0]

                image_array = np.nan_to_num(image_array, nan=0.0, posinf=np.max(image_array), neginf=np.min(image_array))

                # Avoid division by zero by ensuring the maximum value is greater than zero
                max_val = np.max(image_array)
                if max_val > 0:
                    # Normalize the image data to 0-255
                    image_array = (image_array / max_val * 255).astype(np.uint8)
                    # image_array = adjust_contrast(image_array)
                else:
                    # Handle the case where the image is completely black or max_val is 0
                    image_array = np.zeros(image_array.shape, dtype=np.uint8)

                # Convert the numpy array to a PIL Image
                image = Image.fromarray(image_array)
                if image.mode != 'L':
                    image = image.convert('L')  # Convert to grayscale if not already

            # Rotate the image
            original_image = image #.rotate(rotation_angle, expand=True)
    else:
        # If the button is disabled, use the default logo
        original_image = Image.open("logo.png")



    # print(rotation_angle)
    # Apply transformations to the original image
    rotated_image = original_image.rotate(rotation_angle, expand=True)

    # Calculate the new width and height after applying the zoom level
    new_width = int(rotated_image.width * zoom_level)
    new_height = int(rotated_image.height * zoom_level)

    # Check if the new width and height are greater than 0
    if new_width > 0 and new_height > 0:
        # Resize the image
        zoomed_image = rotated_image.resize((new_width, new_height))
    else:
        # If the new width or height is <= 0, use the original rotated image
        zoomed_image = rotated_image

    # Convert the PIL image to a Tkinter PhotoImage
    new_image = ImageTk.PhotoImage(zoomed_image)

    # Update the canvas image
    canvas.itemconfig(image_on_canvas, image=new_image)
    canvas.image = new_image  # Keep a reference to avoid garbage collection



if sys.platform == "win32":
    def zoom(event):

        global zoom_level
        if event.delta > 0:
            zoom_level *= 1.1  # Zoom in
        else:
            zoom_level /= 1.1  # Zoom out

        update_image()
elif sys.platform == "linux":
    def zoom(event):
        global zoom_level
        if event.num == 4:  # Scroll up, zoom in
            zoom_level *= 1.1
        elif event.num == 5:  # Scroll down, zoom out
            zoom_level /= 1.1

        update_image()



def rotate(event):
    global rotation_angle
    # print(rotation_angle)
    if rotation_angle == 270:
        rotation_angle = 0
    else:
        rotation_angle += 90  # Rotate clockwise by 10 degrees
    update_image()


###column0######
Title_label = tk.Label(root,text='SATORI')
Title_label.pack(padx=20,pady=20)
Title_label.config(font=('courier',44))
Title_label.grid(column=0,row=0, sticky="w")


btn_open = tk.Button(root, text='Open folder',command=open_dir, height = 1,  width = 20 )
btn_open.config(font=('courier',12))
btn_open.grid(column=0,row=1, sticky="w")

count_files = tk.StringVar()
count_files.set('Total Number of Files: 0')
count_label = tk.Label(root,textvariable=count_files)
count_label.config(font=('courier',12))
count_label.grid(column=0,row=1, sticky="s")

lb = ttk.Treeview(root, columns=('Image','idx','File Name'))
lb.heading('#0', text='Image')
lb.heading('#1', text='idx')
lb.heading('#2', text='File Name')
lb.column('#0', stretch=tk.NO, width=100, anchor=tk.W)
lb.column('#1', stretch=tk.NO, width=35, anchor=tk.CENTER)  # Adjust column width as needed
lb.column('#2', stretch=tk.NO, width=100, anchor=tk.W)



lb.grid(column=0, row=2, columnspan=1, rowspan=2, sticky="nsew")
lb.bind("<<TreeviewSelect>>", callback)
style = ttk.Style()
style.configure("Treeview", rowheight=50)



# Create a Scrollbar and associate it with the Treeview
scrollbar = ttk.Scrollbar(root, orient="vertical", command=lb.yview)
lb.configure(yscrollcommand=scrollbar.set)

# Position the Treeview and the Scrollbar in the window
scrollbar.grid(column=1, row=2, rowspan=2, sticky='wns')

# Function to handle mouse scroll
def mouse_scroll(event):
    if event.delta:
        lb.yview_scroll(int(-1*(event.delta/120)), "units")
    else:
        if event.num == 5:
            move = 1
        else:
            move = -1
        lb.yview_scroll(move, "units")

# Binding mouse scroll events
lb.bind("<MouseWheel>", mouse_scroll)  # For Windows and MacOS

textBox = tk.Entry(root, width=40, textvariable=Filter_Key,font=('courier',12))
textBox.configure(justify=tk.CENTER)
textBox.grid(column=0,row=4,columnspan=1, sticky="nw", pady=(10, 10))

btn_info = tk.Button(root, text='header info',command=show_img_info, height = 1,  width = 20 )
btn_info.config(font=('courier',12))
btn_info.grid(column=0,row=4,columnspan=1, sticky="se", pady=(5, 5))

btn_filter = tk.Button(root, text='Filter',command=Filter_Files, height = 1,  width = 10 )
btn_filter.config(font=('courier',12))
btn_filter.grid(column=0,row=4,columnspan=1, sticky="ne", pady=(5, 5))

btn_sorter = tk.Button(root, text='Sort Files',command=mass_Filter, height = 1,  width = 20 )
btn_sorter.config(font=('courier',12))
btn_sorter.grid(column=0,row=4,columnspan=1, sticky="sw", pady=(5, 5))


mode_var = tk.IntVar(value=1)
tk.Radiobutton(root, text="mode single", variable=mode_var, value=1).grid(column=0,row=5, sticky=tk.NW, padx=5, pady=5)
tk.Radiobutton(root, text="mode all   ", variable=mode_var, value=2).grid(column=0,row=5, sticky=tk.N, padx=5, pady=5)

###column1######
select_img = tk.StringVar()
select_img.set("")
img_name_label = tk.Label(root,textvariable= select_img)
img_name_label.config(font=('courier',9))
img_name_label.grid(column=1,row=0,sticky="w")

canvas = tk.Canvas(width=800, height=600, bg='black')

canvas.grid(column=1,row=1,columnspan=2,rowspan=3, sticky="")

filename = tk.PhotoImage(file ="logo.png" )
image_on_canvas = canvas.create_image(800/2, 600/2, anchor=tk.CENTER, image=filename)

trackbar = tk.Scale(root, from_=1, to=40 ,length=800,orient=tk.HORIZONTAL,command=update_canvas,digits=1,resolution=0.23)
trackbar.config(state="disabled")
trackbar.grid(column=1,row=4,sticky="w",columnspan=3)


btn_go = tk.Button(root, text='<< GO! >>',command= pred_init, height = 1,  width = 80  ) #
btn_go.config(font=('courier',12),state="disabled")
btn_go.grid(column=1,row=5,rowspan=2,sticky="w")

# Bind mouse events to canvas
# canvas.bind("<MouseWheel>", zoom)
if sys.platform == "win32":
    canvas.bind("<MouseWheel>", zoom)
elif sys.platform == "linux":
    canvas.bind("<Button-4>", zoom)
    canvas.bind("<Button-5>", zoom)
canvas.bind("<Button-3>", rotate)  # Right-click to rotate

# Initial update of the canvas image
update_image()

###column3######
btn_close = tk.Button(root, text='Close',command= close_window, height = 1,  width = 20) #
btn_close.config(font=('courier',12))
btn_close.grid(column=5,row=5, sticky="w")


btn_Save = tk.Button(root, text='Save',command= copy_images, height = 1,  width = 20) #
btn_Save.config(font=('courier',12))
btn_Save.grid(column=4,row=5, sticky="w")


img_details = tk.StringVar()
img_details.set("[info]:")
detail_label = tk.Label(root,textvariable= img_details, justify=tk.LEFT)
detail_label.config(font=('courier',12))
detail_label.grid(column=4,row=2,columnspan=2,sticky="nw")


img_area = tk.StringVar()
img_area.set("AAo:"+"\n"+
             "DAo:"+"\n"+
             "AAo_Max:"+"\n"+
             "DAo_Max:"+"\n"+
             "AAo_Min:"+"\n"+
             "DAo_Min:"+"\n")
Area_label = tk.Label(root,textvariable= img_area, justify=tk.LEFT)
Area_label.config(font=('courier',12))
Area_label.grid(column=4,row=3,columnspan=1,sticky="w")

AD =tk.StringVar()
AD.set("AAo_AD: \nDAo_AD:")
AD_label= tk.Label(root,textvariable= AD, justify=tk.LEFT)
AD_label.config(font=('courier',12))
AD_label.grid(column=5,row=4,columnspan=1,sticky="ws")


def only_numbers(char):
    return char.isdigit()
validation = root.register(only_numbers)



label= tk.Label(root,text="Enter Pulse Pressure:", justify=tk.LEFT)
label.config(font=('courier',12))
label.grid(column=4,row=3,columnspan=1,sticky="s")

pp = tk.IntVar()
textBox = tk.Entry(root, width=34,validate="key", validatecommand=(validation, '%S'),textvariable=pp, font=('courier',12))
textBox.configure(justify=tk.CENTER)
textBox.grid(column=5,row=3, sticky="s")

def calculate():
    if btn_go['state'] == 'disabled':
        tk.messagebox.showwarning("Warning", "Warning select image!")
        exit()
    if pp.get() == None:
        pass
    elif pp.get() == 0:
        tk.messagebox.showwarning("Warning", "Warning your patient is dead!")
    else:

        AAoMax = float(img_area.get().split("\n")[2].replace("AAo_Max:",""))
        AAoMin = float(img_area.get().split("\n")[4].replace("AAo_Min:",""))

        DAoMax = float(img_area.get().split("\n")[3].replace("DAo_Max:",""))
        DAoMin = float(img_area.get().split("\n")[5].replace("DAo_Min:",""))

        AAo = cal_ad(AAoMax,AAoMin,pp.get())
        DAo = cal_ad(DAoMax,DAoMin,pp.get())

        AD.set("AAo_AD:"+'{0:.3f}'.format(AAo*1000)+"10\u207b\u00b3 mmHg⁻¹"+" \nDAo_AD:"+'{0:.3f}'.format(DAo*1000)+"10\u207b\u00b3 mmHg⁻¹")


btn_calculate =tk.Button(root, text='Calculate AD',command=calculate, height = 1,  width = 20 )
btn_calculate.config(font=('courier',12))
btn_calculate.grid(column=4,row=4, sticky="w")

credit_label = tk.Label(root,text="Creator   : Tuan Aqeel Bohoran\nSupervisor:Archontis Giannakidis \n[Nottingham Trent University]", justify=tk.LEFT)
credit_label.config(font=('courier',9))
credit_label.grid(column=0,row=5,columnspan=1,sticky="s")


unit_label = tk.Label(root,textvariable= img_area, justify=tk.LEFT)

img_unit = tk.StringVar()
img_unit.set("mm\u00b2"+"\n"+
             "mm\u00b2"+"\n"+
             "mm\u00b2"+"\n"+
             "mm\u00b2"+"\n"+
             "mm\u00b2"+"\n"+
             "mm\u00b2"+"\n")
Area_label = tk.Label(root,textvariable= img_unit, justify=tk.LEFT)
Area_label.config(font=('courier',12))
Area_label.grid(column=5,row=3,columnspan=1,sticky="w")

img_unit_top = tk.StringVar()
img_unit_top.set("")

unit_label = tk.Label(root,textvariable= img_unit_top, justify=tk.LEFT)
unit_label.config(font=('courier',12))
unit_label.grid(column=5,row=2,columnspan=1,sticky="nw")




root.mainloop()
