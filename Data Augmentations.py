import Augmentor
from Augmentor.Operations import Operation
import PIL
import os, errno, shutil


## Data Augmentation
class Noise(Operation):
    def __init__(self, probability, noise_type='gauss'):
        Operation.__init__(self, probability)
    # 'gauss'    --> Gaussian-distributed additive noise
    # 'poisson'  --> Poisson-distributed noise
    # 's&p'      --> Replaces random pixels with 0 or 1.
    # 'speckle'  --> Multiplicative noise using out = image + n*image, 
    #                where 'n' is uniform noise with specified mean and variance.
    
    def perform_operation(self, images, noise_type = 'gauss'):
        
        def do(image, noise_type):
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if noise_type == 'gauss':
                row, col, ch = image.shape[0:3]
                mean = 0
                var  = 0.1
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row,col,ch))
                gauss = gauss.reshape(row,col,ch)
                noisy = image + gauss
                
            elif noise_type == 's&p':
                row, col, ch = image.shape[0:3]
                s_vs_p = 0.5
                amount = 0.004
                noisy  = np.copy(image)
                # Salt mode
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [np.random.randint(0,i-1, int(num_salt)) for i in image.shape]
                noisy[coords] = 1
                
                # Pepper mode
                num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
                coords = [np.random.randint(0,i-1, int(num_pepper)) for i in image.shape]
                noisy[coords] = 0
                
            elif noise_type == 'poisson':
                vals = len(np.unique(image))
                vals = 2 ** np.ceil(np.log2(vals))
                noisy = np.random.poisson(image * vals) / float(vals)
            
            elif noise_type == 'speckle':
                row, col,ch = image.shape
                gauss = np.random.randn(row,col,ch)
                gauss = gauss.reshape(row, col,ch)
                noisy = image + image*gauss
            
            image = PIL.Image.fromarray(image)
            return image
        augmented_images = []
        for image in images:
            augmented_images.append(do(image,noise_type))
        return augmented_images
    
class Blur(Operation):
    def __init__(self, probability, blur_type='mean'):
        Operation.__init__(self, probability)
    
    def perform_operation(self, images, blur_type='mean'):
        
        def do(image):
            image = np.array(image)
            if blur_type == 'mean':
                image = cv2.blur(image, (7,7))
            elif blur_type =='gauss':
                image = cv2.GaussianBlur(image, (7,7), cv2.BORDER_DEFAULT)
            elif blur_type == 'median':
                image = cv2.medianBlur(image, 5)
            elif blur_type == 'bilateral':
                image = cv2.bilateralFilter(image, 9,15,15)
            
            image = PIL.Image.fromarray(image)
            return image
        augmented_images = []
        for image in images:
            augmented_images.append(do(image))
        return augmented_images
                

def augment_images(source_dir = None, output_dir=None,
                   sample_size = 10000):
    
    shutil.rmtree(output_dir, ignore_errors=True)
    
    pipeline = Augmentor.Pipeline(source_directory= source_dir,
                                  output_directory= output_dir,
                                  save_format='jpg')

    pipeline.rotate(probability = 0.1,
                max_left_rotation=3,
                max_right_rotation=3)
    
    pipeline.random_distortion(probability=0.1,
                               grid_width=4,
                               grid_height=4,
                               magnitude=1)
    
    pipeline.gaussian_distortion(probability=0.1,
                                 grid_width=4,
                                 grid_height=4,
                                 magnitude=1,
                                 method='in',
                                 corner='bell')
    
    pipeline.greyscale(probability=0.1)
    
    pipeline.histogram_equalisation(probability=0.1)
    
    pipeline.random_color(probability=0.1,
                          min_factor=0.4,
                          max_factor=0.5)
    
    pipeline.random_contrast(probability=0.1,
                             min_factor=0.4,
                             max_factor=0.5)
    
    pipeline.random_brightness(probability=0.1,
                               min_factor=0.4,
                               max_factor=0.5)
    
    pipeline.zoom(probability = 0.5,
                  min_factor  = 1.1,
                  max_factor  = 1.5)
    
    pipeline.flip_left_right(probability=0.4)
    
    for method in ['gauss', 's&p', 'poisson', 'speckle']:
        pipeline.add_operation(Noise(probability=0.3, noise_type=method))
    
    for method in ['mean', 'gauss', 'median', 'bilateral']:
        pipeline.add_operation(Blur(probability=0.3, blur_type=method))
    
    pipeline.sample(sample_size, multi_threaded=False)
    
    
def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
indir = 'directory'

for folder in os.listdir(indir):
    augmented_dir = os.path.join("directory", folder)
    make_dir(augmented_dir)
    for img in os.listdir(os.path.join(indir, folder)):
        im = PIL.Image.open(os.path.join(indir, folder, img))
        try:
            im = im.convert("RBG")
        except:
            pass
#         print(im)
#         os.remove(os.path.join(indir,file, img))
#         im.save(os.path.splitext(os.path.join(indir, folder, img))[0])
        
    augment_images(source_dir=os.path.join(indir, folder), output_dir=os.path.abspath(augmented_dir), sample_size =600)
