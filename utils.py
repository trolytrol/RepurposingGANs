import numpy as np
from PIL import Image
import torch, os

def get_stack(inst_gen_model, latent, stack_size):
    '''
    inst_gen_model = Instrumented gen_model
    latent
    '''
    with torch.no_grad():
        img = inst_gen_model(latent)

        features = inst_gen_model.retained_features()
        all_layers = list(features.keys())
        stacks = None
        for key in all_layers:
            value = features[key][0].unsqueeze(0)
            if value.shape[2] < stack_size:
                md = 'bilinear' # Upsample
                ac = True # Align_corner
            else:
                md = 'area' # Downsample
                ac = None # Align_corner
            resized = F.interpolate(value, size=(stack_size, stack_size),
                                    mode=md, align_corners=ac)
            if stacks is None:
                stacks = resized
            else:
                stacks = torch.cat((stacks, resized), dim=1)
    #stacks : [sum, h, w]
    return stacks, img

def z_sample(depth=512, seed=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, depth) # [b, component]
    result = torch.from_numpy(z).float()
    return result

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

def load_celeba_mask(mask_path, mask_name, mask_size, n_class):
    '''
    Load and combine square mask
    '''
    parts_4 = {'eye' : ['_l_eye', '_r_eye'],
            'lip' : ['_l_lip', '_u_lip', '_mouth'],
            'nose' : ['_nose']
        }
    parts_10 = {
            'hair' : ['_hair'],
            'eyebrow' : ['_l_brow', '_r_brow'],
            'eye' : ['_l_eye', '_r_eye'],
            'lip' : ['_l_lip', '_u_lip', '_mouth'],
            'nose' : ['_nose'],
            'skin' : ['_skin'],
            'ear' : ['_l_ear', '_r_ear'],
            'neck' : ['_neck'],
            'clothes' : ['_cloth']
        }
    if n_class==4:
        parts = parts_4
    elif n_class==10:
        parts = parts_10

    total_mask = np.zeros((mask_size, mask_size, n_class))
    for i, part in enumerate(parts.keys()):
        part_mask = np.zeros((mask_size, mask_size, 1))
        for subpart in parts[part]:
            if os.path.exists(os.path.join(mask_path, mask_name + subpart + '.png')):
                mask = Image.open(os.path.join(mask_path, mask_name + subpart +'.png')).convert('L').resize((mask_size,mask_size))
                mask.load()
                mask = np.asarray(mask, dtype='uint8')
                mask = (mask/255).astype(np.float)
                mask[mask>=0.4] = 1
                mask[mask<0.4] = 0
                mask = np.expand_dims(mask, axis=2)
            else:
                mask = np.zeros((mask_size, mask_size, 1)).astype(np.float)
            part_mask[:,:,:] += mask[:,:,:]
        part_mask[part_mask>=1] = 1
        total_mask[:,:,i] = part_mask[:,:,0]

    total_mask[:,:,n_class-1] = 1 - np.sum(total_mask[:,:,0:n_class-1], axis=2)

    total_mask = np.reshape(total_mask, (-1, n_class))
    total_mask = torch.from_numpy(np.argmax(total_mask, axis=1))
    total_mask = F.one_hot(total_mask, num_classes=n_class).numpy()
    total_mask = np.reshape(total_mask, (mask_size, mask_size, n_class))

    total_mask = np.expand_dims(total_mask, axis=0)
    return total_mask

def load_mask(mask_path, mask_name, mask_class):
    horse_dict = {
        0: (255, 255, 0), #head
        1: (255, 0, 0), #body
        2: (0, 0, 255), #leg
        3: (0, 255, 0), #tail
        4: (0, 0, 0) #bg
    }
    car_dict = {
        0: (255, 255, 0), #Wheel
        1: (28, 230, 255), #Window
        2: (255, 52, 255), #Body
        3: (0, 0, 0), #BG
        4: (255, 74, 70), #Light
        5: (0, 137, 65) #PLate
    }
    if mask_class == "horse":
        cls_dict = horse_dict
    elif mask_class == "car":
        cls_dict = car_dict
    else:
        raise NotImplementedError

    img = Image.open(os.path.join(mask_path, mask_name + '.png'))
    img.load()
    img = np.array(img)
    oneh = rgb_to_onehot(img, cls_dict)
    mask = np.expand_dims(oneh, axis=0)
    mask = mask[:,:,:,1:]
    return mask

def viz(label, seg_size):
    '''
    (In) label : (H*W, 1)
    (Out) : (seg_size, seg_size, 3)
    '''
    result = np.zeros((seg_size * seg_size, 3), dtype=np.uint8)
    for pixel in range(len(label)):
        result[pixel] = high_contrast_arr[label[pixel]]
    result = result.reshape((seg_size, seg_size, 3))
    return result

high_contrast = [
    [255, 255, 0], [28, 230, 255], [255, 52, 255], [0, 0, 0],
    [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89],
    [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172],
    [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135],
    [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0],
    [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128],
    [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160],
    [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0],
    [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153],
    [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111],
    [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191],
    [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9],
    [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255]]
high_contrast_arr = np.array(high_contrast, dtype=np.uint8)

