import numpy as np
import cv2
import matplotlib.pyplot as plt


def fun(prev_image,curr_image):

    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,curr_gray, flow = None , pyr_scale = 0.5 ,levels = 3,winsize =15, iterations = 3 ,poly_n = 5, poly_sigma = 1.2 ,flags = 0 )
    spacing = 20

    h,w, *_ = flow.shape
    nx = int(w/spacing)
    ny = int(h/spacing)

    x = np.linspace(0,w-1,nx,dtype=np.int64)
    y = np.linspace(0,h-1,ny,dtype=np.int64)

    flow_notdense = flow[np.ix_(y,x)]
    u = flow_notdense[:,:,0]
    v = flow_notdense[:,:,1]

    uv_mag = np.sqrt(u**2+v**2)
    u[uv_mag<3] = np.NaN
    v[uv_mag<3] = np.NaN

    fig, axs = plt.subplots(1,1,figsize=(10,10))
    kwargs = {**dict(angles="xy",scale_units = "xy")}
    axs.quiver(x,y,u,v,**kwargs,scale = 1)
    axs.set_ylim(sorted(axs.get_ylim(),reverse=True))
    axs.set_aspect("equal")

    axs.imshow(curr_image)
    print("asdasdads")
    plt.show()
prev_image = cv2.imread("3.jpg")
curr_image = cv2.imread("4.jpg")
fun(curr_image,prev_image)