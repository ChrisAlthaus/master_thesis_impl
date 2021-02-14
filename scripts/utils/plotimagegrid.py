import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


fig = plt.figure(figsize=(4.0, 4.0))
#fig.suptitle(title, y=0.9, fontsize=5)
fig.tight_layout()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(nrows, ncolumns),  # creates 2x2 grid of axes
                axes_pad=0.15,  # pad between axes in inch.
                share_all=True
                )   
basewidth = 400
images = []
for rid, filename, cdist in zip(clusterdata['ids'], clusterdata['imagenames'], clusterdata['cdistances']):
    if drawkpts:
        filepath = os.path.join(imgdir, '{}_overlay.jpg'.format(filename))
    else:
        filepath = os.path.join(imgdir, '{}.jpg'.format(filename))
    img = Image.open(filepath)
    #print(rid, img.size)
    #img = resizeimage(img)
    #print(img.size)
    images.append(np.array(img))

c_added = 0
"""
for ax, im, rid, cdist in zip(grid, images, clusterdata['ids'], clusterdata['cdistances']):
    # Iterating over the grid returns the Axes.
    ax.axis('off')
    imtitle = 'r={} d={:0.2f}'.format(rid, float(cdist))
    #ax.set_title(imtitle, fontdict=None, loc='center', color = "k", y=-0.01)
    ax.imshow(im)
    
    #ax.set_xlabel(imtitle)
    ax.text(0.5,-0.1, imtitle, size=2, ha="center", transform=ax.transAxes)
    c_added = c_added + 1"""
_, axs = plt.subplots(nrows, ncolumns, figsize=(12, 12))
axs = axs.flatten()
for img, ax, rid, cdist in zip(images, axs, clusterdata['ids'], clusterdata['cdistances']): 
    #ax.set_anchor('NW')
    ax.imshow(img)
    #ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    imtitle = 'r={} d={:0.2f}'.format(rid, float(cdist))
    ax.set_xlabel(imtitle, labelpad = 4, fontsize=12) #fontweight='bold'
    c_added = c_added + 1
plt.subplots_adjust(wspace=0.05, hspace=0.12)
#if c_added<nrows*ncolumns:
#    for i in range(c_added,nrows*ncolumns):
#        grid[i].axis('off')
#        ax.text(0.5,-0.1, "Image", size=2, ha="center", transform=ax.transAxes)
plt.savefig(savepath, dpi=400, bbox_inches='tight', pad_inches=0.01)
plt.clf()