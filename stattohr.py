#######################
#?## USER VARIABLES ###
#?#####################

#// Square this because I'll be lazily not-square-rooting later
chroma_similarity = 10**2

video_width = 1920
video_height = 1080

capture_device = 0

buffer_size = 1024

#?######################
#?## /USER VARIABLES ###
#?######################

#? A few other variables
display_mode = 'normal'
fps_display = True

processing_mode = 'average_ppg'

#? Libraries
import cv2 as cv                        # OpenCV                            #* media processing (e.g., r/w, feat detection, manipulation)
import numpy as np                      # Numpy                             #* numerical computing
import cupy as cp                       # CuPy                              #! GPU-accelerated numerical computing                  // PyTorch, TensorFlow, ...
import cusignal                         # CuPy's signal processing library  #! (e.g., noise removal, filter, edge detection)        // ^, SciPy, scikit-image, ...
import cupyx                            # CuPy's scikit-image equivalent    #! additional image processing functions                // ^ , scikit-image, ...
import time                             # Time                              #* timing
import csv                              # CSV                               #* reading and writing CSV files
import random                           # Random                            #* random number generation
import cupyx.scipy.ndimage as cuimg     # CuPy's scikit-image equivalent    #! additional image processing functions                // Torchvision (PyTorch), TensorFlow Image, ...
import matplotlib.pyplot as plt         # Matplotlib                        #* plotting
import matplotlib                       # Matplotlib                        #* plotting
import os                               # OS                                #* operating system functions (r/w files/dir)
import argparse                         #? Set up command line argument to process a file rather than live video: 

#?: ...
parser = argparse.ArgumentParser() 
parser.add_argument("-f", "--file", dest="filename",
                    help="Process a video file rather than a capture device.", metavar="FILE")
# This argument currently doesn't do anything, I wanted to make this user-friendly but never got around to it!
parser.add_argument('-w','--welch', dest='welch_flag', action=argparse.BooleanOptionalAction,
                    help='Compute heart rate using the Welch estimator.')
args = parser.parse_args()

#? Write some data to files
def csv_xyz(filename, data, names):         #* Write a 3D array to a CSV file
    csv_begin(filename, names)                  # Write the column names
    csv_append(filename, data)                  # Write the data
#
def mouseRGB(event, x, y, flags, params):   #* Mouse callback function: print RGB value of the clicked location
    global skin_chroma                                          # Make skin_chroma a global variable
    if event == cv.EVENT_LBUTTONDOWN:                           # If the left mouse button is clicked: ...
        skin_chroma = cp.array(cv.cvtColor(np.array([[frame[y,x]]]), cv.COLOR_BGR2YUV), dtype=cp.float32)[0,0,1:3]      # set the chroma value to the pixel clicked
        print('RGB = ', frame[y,x], 'chroma = ', skin_chroma)                                                           # print the RGB and chroma values
#
def chroma_key(frame, chroma):                                       #* Chroma keying function
    key = frame[:,:,1:3] - chroma                                       # Subtract the chroma value from the frame
    key = cp.less(cp.sum(cp.square(key), axis=2), chroma_similarity)    # Sum the squares of the differences and compare to the similarity threshold
    return key  
#
def chroma_key_display(frame, chroma):                               #* Display chroma key (for convenience/debugging)
    #// """
    #// Convenience function to display the chroma key
    #// """
    key = chroma_key(frame, chroma)                                     # Get the chroma key
    return cp.asnumpy(key*255).astype(np.uint8)                         # Return the key as a uint8 array
#
def moving_average(a, n=3, axis=None):                     #* Moving average function
    if axis is not None:                                        # If the axis is not None, flatten the array (by swapping the axis to the end)
        ret = np.swapaxes(a, 0, axis)
    else:                                                       # Otherwise, just flatten the array
        ret = a
    
    ret = cp.cumsum(ret, axis=axis)                             # take the cumulative sum of the input vector
    ret[n:,...] = ret[n:,...] - ret[:-n,...]                    # subtract the cumsum, offset by n, to get the moving average via kludge
    #// Concatenate together 0 ..the numbers... 0 0 to pad it to the original length
    ret = cp.concatenate((                                      # Concatenate the following arrays:
        #// Following what R does, return fewer 0s at the start if n is even...
        cp.zeros((int(np.floor((n-1)/2)), *ret.shape[1:])),         # (following what R does) return fewer 0s at the start if n is even...
        #// ...then some numbers...
        ret[(n - 1):,...] / n,                                      # ...then the moving average...
        cp.zeros((int(np.ceil((n-1)/2)), *ret.shape[1:]))           # ...then more 0s at the end if n is even (both equal if odd!)
    ))
    
    if axis is not None:                                        # Swap the axis back if we swapped it at the start
        ret = np.swapaxes(ret, 0, axis)

    return ret                                                  # Return the moving average
#
def average_keyed(frame, key):              #* Average the YUV of the pixels which are True in key
    #// """
    #// Return the average YUV of the pixels which are True in key.
    #// Args:
    #//     frame: a cupy array containing the frame
    #//     key: a cupy array of booleans
    #// Returns:
    #//     A cupy array of [Y, U, V]
    #// """
    output = cp.mean(frame[key], axis=0)        # Take the mean of the frame where key is true
    return output                               # Return the mean
#
def csv_begin(filename, data):                      #* Write the column names to a CSV file
    with open(filename, 'w', newline='') as f:          # Open the file so we can write to it
        writer = csv.writer(f)                          # Create a CSV writer...
        writer.writerow(data)                           # ...then write the data
#
def csv_append(filename, data):                     #* Append data to a CSV file
    with open(filename, 'a', newline='') as f:          # Open the file so we can append to it
        writer = csv.writer(f)                          # Create a CSV writer...
        writer.writerows(data)                          # ...then write the data
#
def magnify_colour_ma(ppg, delta=50, n_bg_ma=60, n_smooth_ma=3):    #* Magnify the PPG signal
    ppg = ppg - moving_average(ppg, n_bg_ma, 0)                         # Subtract the moving average of the PPG from the PPG ("Remove slow-moving background component")
    ppg = moving_average(ppg, n_smooth_ma, 0)                           # Smooth the PPG ("Smooth the resulting PPG")
    ppg = cp.nan_to_num(ppg)                                            # Replace NaNs with zeros
    return delta*ppg/cp.max(cp.abs(ppg))                                # Return the PPG, normalised by the biggest deviation ("Make it have a max delta of delta by normalising by the biggest deviation")
#
def magnify_colour_ma_masked(ppg, mask, delta=50, n_bg_ma=60, n_smooth_ma=3):   #* Magnify the PPG signal
    ppg = ppg - moving_average(ppg, n_bg_ma, 0)                                     # Subtract the moving average of the PPG from the PPG ("Remove slow-moving background component")
    mask = moving_average(mask, n_bg_ma, 0)                                         # Smooth the mask
    ppg = moving_average(ppg, n_smooth_ma, 0)                                       # Smooth the resulting PPG
    mask = moving_average(mask, n_smooth_ma, 0)                                     # Smooth the mask
    # Expand the mask to allow it to be used to, er, mask the ppg
    # Remove any pixels in ppg that go to zero at any point in the windows, found because the mask which has been equivalently moving-averaged above drops below 1
    ppg = np.where(mask[:,:,:, cp.newaxis] == 1., ppg, cp.zeros_like(ppg))          # Mask the PPG
    ppg = cp.nan_to_num(ppg)                                                        # Replace NaNs with zeros
    ppg[:,:,:,0] = 0                                                                # Set the Y component to 0
    return delta*ppg/cp.max(cp.abs(ppg))                                            # Return the PPG ("Return the PPG, normalised by the biggest deviation")
#
def Welch_cuda(filename, bvps, fps, nfft=8192):   #* Compute the Welch estimator of heart rate
    #// """
    #// This function computes Welch's method for spectral density estimation on CUDA GPU.
    #// Args:
    #//     bvps(float32 cupy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
    #//     fps (cupy.float32): frames per seconds.
    #//     minHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    #//     maxHz (cupy.float32): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
    #//     nfft (cupy.int32): number of DFT points, specified as a positive integer.
    #// Returns:
    #//     Sample frequencies as float32 cupy.ndarray, and Power spectral density or power spectrum as float32 cupy.ndarray.
    #// """
    bvps = cp.transpose(cp.array(bvps, dtype=cp.float32))  # Transpose the BVP signal
    for i in range(128, bvps.shape[0]-128):                #* LOOP through the BVP signal: ...
        print(i, bvps.shape)                                    # ... print the current frame number and the shape of the BVP signal
        t = (i+128)/fps                                         # ... calculate the time
        #// -- periodogram by Welch
        F, P = cusignal.welch(bvps[(i-128):(i+128)], nperseg=256,   # ... compute the Welch periodogram
                                noverlap=200, fs=fps, nfft=nfft)
                                                                                            #* write t, F, P to our CSV: ...
        #// This awful expression is needed:
        #// * We have F, a long 1D tuple of frequencies and P, a 2D tuple for each chroma
        #//   component at each frequency. We also want to add a timestamp column.
        #// * Create an array full of the current time the same length as F with cp.full
        #// * The suffix [:,None] is needed for the 1D t and F arrays because unless they
        #//   are fake-2D, they won't concatenate.
        #// * P needs to be transposed, hence P.T
        print(cp.full((F.shape), t)[:,None].shape, F[:,None].shape, P.T.shape)                  # Debugging
        towrite = cp.concatenate((cp.full((F.shape), t)[:,None],F[:,None],P[:,None]), axis=1)   # Concatenate the time, frequency, and power
        csv_append(filename, towrite)                                                           # Append the data to the CSV file
#
def display_image(img, window='StattoBPM'):                                                         #* Display an image
    if len(img.shape) == 3:                                                                             # IF the image is 3D: ...
        cv.imshow(window, cv.cvtColor(cp.asnumpy(img).astype(np.uint8), cv.COLOR_YUV2BGR))                  # ... display the image
    elif len(img.shape) == 2:                                                                           # IF the image is 2D: ...
        cv.imshow(window, cp.asnumpy(img).astype(np.uint8)*255)                                             # ... display the image
#
def keypress_action(keypress):                              #* Respond to keypresses
    global display_mode, fps_display
    if keypress != -1:                                          # IF the keypress is in the dictionary of display modes... ("Default keypress is -1, which means 'do nothing' so skip the below")
        if keypress == ord('a'):                                    # IF 'a' is pressed, toggle the alpha channel display mode ("If the keypress is in the dictionary of display modes")
            if display_mode != 'alpha_channel':                         # if it's not already in that mode, put it in that mode
                display_mode = 'alpha_channel'  
            else:                                                       # if it is already in that mode, put things back to normal
                display_mode = 'normal'
            print('Display mode ' + display_mode + ' activated')    # print the display mode
        elif keypress == 48:                                        # IF '0' is pressed, record an in breath
            csv_append('data/breaths.csv', [[t, 0]])
        elif keypress == 46:                                        # IF '.' is pressed, record an out breath
            csv_append('data/breaths.csv', [[t, 1]])
        elif keypress == ord('f'):                                  # IF 'f' is pressed, toggle (enable/disable) the fps display
            fps_display = not(fps_display)
            print('FPS display set to', fps_display)
        elif keypress==27 or keypress == ord('q'):                  # IF 'Esc' or 'q' is pressed, exit the program
            print('Goodbye!')
            exit()
        else:                                                       # ELSE: if any other key is pressed, print the keypress
            print('You pressed %d (0x%x), LSB: %d (%s)' % (keypress, keypress, keypress % 256,
                    repr(chr(keypress%256)) if keypress%256 < 128 else '?'))
#
#? If no filename was specified, open a capture device
if(args.filename is None):              #* IF no filename was specified: ...
    vid = cv.VideoCapture(capture_device)   # Open the capture device ("define a video capture object")
    if not vid.isOpened():                  # IF the capture device is not opened: ...
        print('Error: Cannot open camera')    # ... print an error message
        exit()                                # ... and exit the program
    else:                                   # ELSE: if the capture device is opened: ... print the frame size
        print('Initialising camera...')      
        print('  Default frame size: ' + str(vid.get(cv.CAP_PROP_FRAME_WIDTH)) + '×' + str(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
    vid.set(cv.CAP_PROP_FRAME_WIDTH, video_width)   # Set the frame width ("override default frame size for capture device, which is often 640x480")
    vid.set(cv.CAP_PROP_FRAME_HEIGHT, video_height) # Set the frame height
    is_video_file = False                   # Set is_video_file to False (because we're using a capture device, not a file)
else:                                   #* ELSE: if a filename was specified: ...
    vid = cv.VideoCapture(args.filename)    # Open the video file
    if (vid.isOpened() == False):           # IF the video file is not opened: ... print an error message and exit the program
        print("Error opening video file " + args.filename)
        exit()
    fps = vid.get(cv.CAP_PROP_FPS)          # Get the fps from the video, then the total frame count
    total_frames = vid.get(cv.CAP_PROP_FRAME_COUNT) 
    is_video_file = True                   # Set is_video_file to True ("because we're using a file, not a capture device")

#? create empty lists to store the PPG data and the times
ppg_yuv = []    # PPG data
ppg_rgb = []    # PPG data
times  = []     # times



#? Instantiate: 
skin_chroma = cp.zeros(2, dtype=cp.float32) #* skin_chroma
cv.namedWindow('StattoBPM')                 #* display window (to show the BPM [video])
cv.setMouseCallback('StattoBPM', mouseRGB)  #* mouse callback function (to detect mouse clicks)



#? (IF using a video file:) start a loop to set the chroma key prior to processing
print('Displaying random frames. Click to set chroma. Press A to toggle chroma key view and O once you\'re ready to go!')
if is_video_file:                                                                   #* IF this is a video file: ...
    while True:
        random_frame = random.randrange(int(total_frames/4), int(3*total_frames/4))     # Get a random frame somewhere near the middle of the video
        #// print('Displaying frame', random_frame, 'of', int(total_frames))
        vid.set(1, random_frame)                                                        # Set the video to the random frame
        ret, frame = vid.read()                                                         # Read the frame
        frame_cp = cp.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=cp.float32)     # Convert the frame to YUV

        if display_mode == 'alpha_channel':                                             # IF the display mode is 'alpha_channel': ...
            cv.imshow(                                                               
                'StattoBPM',
                chroma_key_display(frame_cp, skin_chroma)                                   # ... display the chroma key
            )
        else:
            cv.imshow(
                'StattoBPM',
                frame                                                                   # ELSE, display the frame
            )
        keypress = cv.waitKey(100)                                                          # Wait for a keypress

        if keypress == ord('o'):                                                            # IF 'o' is pressed: set the video to the first frame and break the loop
            print('OK, chroma value set! Let\'s compute!')
            vid.set(1, 0)                                                                       #// ("Set to the first frame so the whole video gets processed")
            break

        keypress_action(keypress)                                                           # Respond to keypresses



#? 1st LOOP -- Analysis
print('First pass: analysing video...')
i = 0
t0 = time.time()    # Start the timer (by a setting t0 to the curr time)

while True:                                         #* Start the LOOP:
    ret, frame = vid.read()                             # Read the frame (from camera/file)
    times.append(time.time() - t0)                      # Store the curr time in the buffer

    if i > 0:                                           # calculate fps (if not the first pass [to avoid dividing by zero]; ["of either processing or capture depending on device"])
        fps_calc = len(times) / (times[-1] - times[0])

    if not is_video_file:                               # IF using camera (instead of file): fps is given by the above, rather than specified beforehand
        fps = fps_calc                                      # set fps to fps_calc
        t = times[-1]                                       # set t to the last time in the buffer
    else:                                               # ELSE if using a file: time = frame / fps
        t = i/fps

    if i > 0 and fps_display and i % 100 == 0:          # IF not the first pass, AND fps_display is True, AND i is a multiple of 100: print the frame number and fps
        print('Frame', i, 'of', int(total_frames), '  |  FPS:', np.round_(fps_calc, 3))

    if not ret:                                         # IF ret is false, it usually means 'video file is over', but it's an error either way, so exit the loop
        print('Pass 1 complete!')
        break

    frame_cp = cp.array(frame, dtype=cp.float32)                                    # Convert the frame to a float32 array
    frame_yuv = cp.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=cp.float32)    # Convert the frame to YUV
    skin_key = chroma_key(frame_yuv, skin_chroma)                                   # Get the chroma key
    ppg_rgb.append(average_keyed(frame_cp, skin_key))                               # Append the average keyed frame (RGB) to the PPG data
    ppg_yuv.append(average_keyed(frame_yuv, skin_key))                              # Append the average keyed frame (YUV) to the PPG data

    if display_mode == 'alpha_channel':                  # IF the display mode is 'alpha_channel': ...
        cv.imshow(                                          # ... display the chroma key
            'StattoBPM',
            chroma_key_display(frame_yuv, skin_chroma)  
        )
    else:                                                # ELSE: display the frame
        cv.imshow('StattoBPM', frame)  

    keypress_action(cv.waitKey(1))
    i = i + 1



#? Calculations
print('First pass completed. Doing calculations...')

ppg_rgb_ma = magnify_colour_ma(             #* Magnify the PPG signal             #?("'white', averaging YUV")("'white', averaging RGB")
    cp.array(ppg_rgb, dtype=cp.float64),        # PPG data
    delta=1,                                    # delta
    n_bg_ma=90,                                 # n_bg_ma
    n_smooth_ma=6                               # n_smooth_ma
    )
ppg_yuv_ma = magnify_colour_ma(             #* Magnify the PPG signal            #?("'white', averaging YUV")
    cp.array(ppg_yuv, dtype=cp.float64),        # PPG data
    delta=1,                                    # delta
    n_bg_ma=90,                                 # n_bg_ma
    n_smooth_ma=6                               # n_smooth_ma
    )
ppg_w_ma = cp.mean(ppg_rgb_ma, axis=1)      #* Magnify the PPG signal ("'white', averaging")


outdir = 'output-data-' + args.filename     #* Create a directory to store the output data 
counter = 1
mypath = outdir + '-' + str(counter)    

while os.path.exists(mypath):         
    counter += 1
    mypath = outdir + '-' + str(counter)

os.makedirs(mypath)

csv_xyz(os.path.join(mypath, 'ppg-rgb.csv'), cp.asnumpy(cp.array(ppg_rgb, dtype=cp.float64)), ['b', 'g', 'r'])              # Write the PPG data to a CSV file  (RGB)
csv_xyz(os.path.join(mypath, 'ppg-rgb-ma.csv'), cp.asnumpy(cp.array(ppg_rgb_ma, dtype=cp.float64)), ['b', 'g', 'r'])        # Write the PPG data to a CSV file  (RGB -- ma)
csv_xyz(os.path.join(mypath, 'ppg-yuv.csv'), cp.asnumpy(cp.array(ppg_yuv, dtype=cp.float64)), ['y', 'u', 'v'])              # Write the PPG data to a CSV file  (YUV)
csv_xyz(os.path.join(mypath, 'ppg-yuv-ma.csv'), cp.asnumpy(cp.array(ppg_yuv_ma, dtype=cp.float64)), ['y', 'u', 'v'])        # Write the PPG data to a CSV file  (YUV -- ma)

with open(os.path.join(mypath, 'chroma-key.txt'), 'w') as f:        # Write the chroma key to a text file
    f.write(str(skin_chroma))  

matplotlib.use('TKAgg')                                             # Set the backend to TKAgg

def normalise(x):                                                   #* Normalise and return the data
    return (x - cp.min(x))/(cp.max(x)-cp.min(x))    


data0 = {'time': np.array(range(int(total_frames)))/fps,            #* Plot the RGB data -- time, RGB in object
        'red': cp.asnumpy(cp.array(ppg_rgb, dtype=cp.float64)),         # R
        'green': cp.asnumpy(cp.array(ppg_rgb, dtype=cp.float64)),       # G
        'blue': cp.asnumpy(cp.array(ppg_rgb, dtype=cp.float64))}        # B
#
fig, ax = plt.subplots()                                            #* Plot the RGB data (subplots)
ax.plot('time', 'red', data=data0, color='red')                         # time v R
ax.plot('time', 'green', data=data0, color='green')                     # time v G
ax.plot('time', 'blue', data=data0, color='blue')                       # time v B
ax.set_xlabel('time')                                                   # x-axis label
ax.set_ylabel('RGB')                                                    # y-axis label
plt.show() 
#
#
data = {'time': np.array(range(100,int(total_frames)-100))/fps,     #* ???
        'red': cp.asnumpy(ppg_rgb_ma[100:-100:,2]),
        'green': cp.asnumpy(ppg_rgb_ma[100:-100:,0]),
        'blue': cp.asnumpy(ppg_rgb_ma[100:-100:,1])}
#
fig, ax = plt.subplots()
ax.plot('time', 'red', data=data, color='red')
ax.plot('time', 'green', data=data, color='green')
ax.plot('time', 'blue', data=data, color='blue')
ax.set_xlabel('time')
ax.set_ylabel('RGB')
plt.show()
#
#
data2 = {'time': np.array(range(100,int(total_frames)-100))/fps,    #* ??? 2
        'luminance': cp.asnumpy(ppg_yuv_ma[100:-100:,0]),
        'colour-u': cp.asnumpy(ppg_yuv_ma[100:-100:,1]),
        'colour-v': cp.asnumpy(ppg_yuv_ma[100:-100:,2])}
#
fig, ax = plt.subplots()
ax.plot('time', 'luminance', data=data2, color='black')
ax.plot('time', 'colour-u', data=data2, color='green')
ax.plot('time', 'colour-v', data=data2, color='magenta')
ax.set_xlabel('time')
ax.set_ylabel('YUV')
plt.show()



#? Reopen the video
vid = cv.VideoCapture(args.filename)                        # Open the video file

if vid.isOpened() == False:                                 # If the video file is not opened: print err and exit program
    print("Error opening video file " + args.filename)
    exit()
is_video_file = True                                        # bc we're using a file (not a camera)



#? 2nd LOOP -- adding stuff
print('Second pass: saving results!')
frames_path = os.path.join(mypath, 'frames-uvw')    #* Create a directory to store the frames
os.makedirs(frames_path)    
times = []                                          # time cache ("buffer"[?])
i = 0
t0 = time.time()                                    # Start the timer (by setting t0 to the curr time)

while True:                                         #* Start the loop
    ret, frame = vid.read()                             # Read the frame (from camera/file)
    times.append(time.time() - t0)                      # Store the curr time in the buffer

    if i > 0:                                        # calculate fps (if not the first pass [to avoid dividing by zero]; ["of either processing or capture depending on device"])
        fps_calc = len(times) / (times[-1] - times[0])

    if not is_video_file:                             # If using camera (instead of file): fps is given by the above, rather than specified beforehand
        fps = fps_calc
        t = times[-1]
    else:                                             # ELSE if using a file: time = frame / fps
        t = i/fps

    if i > 0 and fps_display and i % 100 == 0:        # IF not the first pass, AND fps_display is True, AND i is a multiple of 100: print the frame number and fps
        print('Frame', i, 'of', int(total_frames), '  |  FPS:', np.round_(fps_calc, 3))

    if not ret:                                       # IF ret is false, it usually means 'video file is over', but it's an error either way, so exit the loop       
        print('Pass 2 complete!')
        break

    #//frame_rgb = cp.array(frame, dtype=cp.float32)
    frame_yuv = cp.array(cv.cvtColor(frame, cv.COLOR_BGR2YUV), dtype=cp.float32)                                                                # Convert the frame to YUV
    skin_key = chroma_key(frame_yuv, skin_chroma)                                                                                               # Get the chroma key
    colours_uv = cp.moveaxis(cp.array([cp.zeros_like(skin_key), skin_key * ppg_yuv_ma[i][1], skin_key * ppg_yuv_ma[i][2]]), 0, -1)              # Create the UV colours (YUV???)
    #//colours_yuv = cp.moveaxis(cp.array([skin_key * ppg_yuv_ma[i][0], skin_key * ppg_yuv_ma[i][1], skin_key * ppg_yuv_ma[i][2]]), 0, -1)
    #//colours_g = cp.moveaxis(cp.array([cp.zeros_like(skin_key), skin_key * ppg_rgb_ma[i][1], cp.zeros_like(skin_key)]), 0, -1)
    colours_w = cp.moveaxis(cp.array([cp.zeros_like(skin_key), skin_key * ppg_w_ma[i], skin_key * ppg_w_ma[i]]), 0, -1)                         # Create the W colours
    # ("Add a bunch of zeros in the Y component" [to make it look like a greyscale image])
    #//output_uv = cv.cvtColor(cp.asnumpy(frame_yuv + colours_uv[0:1080, 0:1920, :]*50000).astype(np.uint8), cv.COLOR_YUV2BGR)
    output_uv_w = cv.cvtColor(cp.asnumpy(frame_yuv + colours_w[0:1080, 0:1920]*10000).astype(np.uint8), cv.COLOR_YUV2BGR)                      # Output the UVW frame
    #//output_yuv = cv.cvtColor(cp.asnumpy(frame_yuv + colours_yuv[0:1080, 0:1920, :]*50000).astype(np.uint8), cv.COLOR_YUV2BGR)
    #//output_g = cp.asnumpy(frame_rgb + colours_g[0:1080, 0:1920, :]*50000).astype(np.uint8)
    cv.imshow('StattoBPM', output_uv_w)
    #//cv.imshow('StattoAdd', cv.cvtColor(cp.asnumpy(colours_uv[0:1080, 0:1920, :]*50000).astype(np.uint8)+128, cv.COLOR_YUV2BGR))
    cv.imwrite(os.path.join(frames_path, 'uvw_magnified-'+f'{i:05}'+'.png'), output_uv_w)
    #//cv.imwrite('output_yuv/macombined-'+f'{i:05}'+'.png', output_yuv)
    #//cv.imwrite('output_rgb/macombined-'+f'{i:05}'+'.png', output_g)
    keypress_action(cv.waitKey(1))
    i = i + 1

#? This is the code you'd need to do the Welch estimate of heart rate (I've commented it out because I did this manually for the video, but it should respond to the command line argument -w!):
#//ppg_w_ma = ppgw_read('output-data-video\AJS_A7S_20220801_2202.mov-1\ppg-rgb-ma.csv')
#//Welch_cuda('welch.csv', ppg_w_ma, 59.94)