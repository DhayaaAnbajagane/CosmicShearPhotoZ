from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as pl
import numpy as np
import pandas
import time

def readCOSMOS():
    '''Function to read the COSMOS input files.
    Returns arrays fluxes,errors,redshifts,counts giving for each unique object
    its ugrizJHK fluxes, flux errors, Laigle redshift, number of Balrog counts,
    and radec array.
    '''
    # Read the master file.  Mag zeropoints are all 30.0
    cosmos = pandas.read_hdf('cosmos.hdf5','fluxes')

    # Pull out numpy arrays for our quantities of interest
    fluxes = np.vstack( [cosmos['BDF_FLUX_DERED_U'],
                         cosmos['BDF_FLUX_DERED_G'],
                         cosmos['BDF_FLUX_DERED_R'],
                         cosmos['BDF_FLUX_DERED_I'],
                         cosmos['BDF_FLUX_DERED_Z'],
                         cosmos['BDF_FLUX_DERED_J'],
                         cosmos['BDF_FLUX_DERED_H'],
                         cosmos['BDF_FLUX_DERED_K']]).transpose()
    errors = np.vstack( [cosmos['BDF_FLUX_ERR_DERED_U'],
                         cosmos['BDF_FLUX_ERR_DERED_G'],
                         cosmos['BDF_FLUX_ERR_DERED_R'],
                         cosmos['BDF_FLUX_ERR_DERED_I'],
                         cosmos['BDF_FLUX_ERR_DERED_Z'],
                         cosmos['BDF_FLUX_ERR_DERED_J'],
                         cosmos['BDF_FLUX_ERR_DERED_H'],
                         cosmos['BDF_FLUX_ERR_DERED_K']]).transpose()
    redshifts = np.array(cosmos['Z'])
    radec = np.vstack([cosmos['RA'],
                         cosmos['DEC']]).transpose()

    # Reduce to the unique COSMOS objects, keep track of number of Balrog detections
    junk,indices,counts = np.unique(fluxes[:,0],return_index=True,return_counts=True)
    fluxes = fluxes[indices]
    errors = errors[indices]
    redshifts = redshifts[indices]
    radec = radec[indices]
    return fluxes, errors, redshifts, counts, radec

def somPlot3d(som,az=200.,el=30.):
    # Make a 3d plot of cells weights in color space.  az/el are plot view angle
    mags = 30. - 2.5*np.log10(som.weights)
    ug = mags[:,0] - mags[:,1]
    gi = mags[:,1] - mags[:,3]
    ik = mags[:,3] - mags[:,7]
    imag = mags[:,3]
    fig = pl.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.azim = az
    ax.elev = el
    ax.scatter(gi,ik,imag,c=ug, cmap='Spectral_r')
    # Draw the outline of the SOM edges
    xx = np.arange(som.shape[0],dtype=int)
    yy = np.arange(som.shape[1],dtype=int)
    xxx = np.hstack( (xx,
                      np.ones(len(yy)-2,dtype=int)*xx[-1],
                      xx[::-1],
                      np.zeros(len(yy)-1,dtype=int)))
    yyy = np.hstack( (np.zeros(len(xx),dtype=int),
                      yy[1:-1],
                      np.ones(len(xx)-1,dtype=int)*yy[-1],
                      yy[-1::-1]))
    ii = np.ravel_multi_index((xxx,yyy),som.shape)
    ax.plot(gi[ii],ik[ii],imag[ii],'k--')
    ax.set_title('Node locations') 
    ax.set_aspect('equal')
    ax.set_xlabel('gi')
    ax.set_ylabel('ik')
    ax.set_zlabel('imag')
    return

def somPlot2d(som):
    # Make a 2d plot of cells weights in color-color diagram space.
    mags = 30. - 2.5*np.log10(som.weights)
    ug = mags[:,0] - mags[:,1]
    gi = mags[:,1] - mags[:,3]
    ik = mags[:,3] - mags[:,7]
    imag = mags[:,3]
    fig = pl.figure(figsize=(6,7))
    # First a color-color plot of nodes
    pl.scatter(gi,ik,c=imag, alpha=0.3,cmap='Spectral')
    # Draw the outline of the SOM edges
    xx = np.arange(som.shape[0],dtype=int)
    yy = np.arange(som.shape[1],dtype=int)
    xxx = np.hstack( (xx,
                      np.ones(len(yy)-2,dtype=int)*xx[-1],
                      xx[::-1],
                      np.zeros(len(yy)-1,dtype=int)))
    yyy = np.hstack( (np.zeros(len(xx),dtype=int),
                      yy[1:-1],
                      np.ones(len(xx)-1,dtype=int)*yy[-1],
                      yy[-1::-1]))
    ii = np.ravel_multi_index((xxx,yyy),som.shape)
    pl.plot(gi[ii],ik[ii],'k-')
    pl.title('Node locations') 
    pl.gca().set_aspect('equal')
    cb = pl.colorbar()
    cb.set_label('imag')
    pl.xlabel('gi')
    pl.ylabel('ik')
    return

def somDomainColors(som):
    # Make 4-panel plot colors and mag across SOM space
    mags = 30. - 2.5*np.log10(som.weights)
    ug = mags[:,0] - mags[:,1]
    gi = mags[:,1] - mags[:,3]
    ik = mags[:,3] - mags[:,7]
    imag = mags[:,3]
    fig = pl.figure(figsize=(6,7))
    
    fig,ax = pl.subplots(nrows=2,ncols=2,figsize=(8,8))
    im = ax[0,0].imshow(gi.reshape(som.shape),interpolation='nearest',origin='lower',
                        cmap='Spectral_r')
    ax[0,0].set_title('gi')
    ax[0,0].set_aspect('equal')
    pl.colorbar(im, ax=ax[0,0])

    im = ax[1,0].imshow(ug.reshape(som.shape),interpolation='nearest',origin='lower',
                        cmap='Spectral_r')
    ax[1,0].set_title('ug')
    ax[1,0].set_aspect('equal')
    pl.colorbar(im, ax=ax[1,0])

    im = ax[0,1].imshow(ik.reshape(som.shape),interpolation='nearest',origin='lower',
                            cmap='Spectral_r')
    ax[0,1].set_title('ik')
    ax[0,1].set_aspect('equal')
    pl.colorbar(im, ax=ax[0,1])

    im = ax[1,1].imshow(imag.reshape(som.shape),interpolation='nearest',origin='lower',
                            cmap='Spectral')
    ax[1,1].set_title('imag')
    ax[1,1].set_aspect('equal')
    pl.colorbar(im, ax=ax[1,1])
    return

def plotSOMz(som, cells, zz, subsamp=1,figsize=(8,8)):
    '''Make 4-panel plot showing occupancy of SOM by a redshift sample and statistics
       of redshift distribution in each cell.'''
    nbins = np.prod(som.shape)
    nn = np.histogram(cells,bins=nbins,range=(-0.5,nbins-0.5))[0]
    zmean = np.histogram(cells,bins=nbins,range=(-0.5,nbins-0.5),weights=zz[::subsamp])[0]/nn
    zvar = np.histogram(cells,bins=nbins,range=(-0.5,nbins-0.5),weights=(zz*zz)[::subsamp])[0]/nn
    zrms = np.sqrt(zvar-zmean*zmean)
    zmed = np.array([np.median(zz[cells==i]) for i in range(nbins)])

    fig,ax = pl.subplots(nrows=2,ncols=2,figsize=figsize)

    im = ax[0,0].imshow(np.log10(nn.reshape(som.shape)),interpolation='nearest',origin='lower')#,
                        #cmap=cmr.heat)
    ax[0,0].set_aspect('equal')
    ax[0,0].set_title('Sources per cell')
    pl.colorbar(im, ax=ax[0,0])

    useful = nn>4
    im = ax[0,1].imshow(zmed.reshape(som.shape),interpolation='nearest',origin='lower',
                            vmax=2.5, vmin=0., cmap='Spectral')
    ax[0,1].set_aspect('equal')
    ax[0,1].set_title('z_median')
    pl.colorbar(im, ax=ax[0,1])

    im = ax[1,0].imshow((zrms/(1+zmed)).reshape(som.shape),interpolation='nearest',origin='lower',
                            cmap='Spectral')
    ax[1,0].set_aspect('equal')
    ax[1,0].set_title('std(z)/(1+zmed)')
    pl.colorbar(im, ax=ax[1,0])

    print('Median sig(ln(z)):',np.median((zrms/(1+zmed))[useful]))

    # make another plot showing rms of neighbor cells
    tmp = zmed.reshape(som.shape)
    tmp2 = np.stack((tmp[:-2,:-2],
                     tmp[:-2,1:-1],
                     tmp[:-2,2:],
               tmp[1:-1,:-2],
               tmp[1:-1,1:-1],
               tmp[1:-1,2:],               
               tmp[2:,:-2],
               tmp[2:,1:-1],
               tmp[2:,2:]), axis=0)             
    grad= np.std(tmp2,axis=0)/tmp[1:-1,1:-1] / 2.
    print('Median neighbor sig(ln(z)):',np.median(grad[~np.isnan(grad)]))
    tmp = (zrms/(1+zmean)).reshape(som.shape)[1:-1,1:-1]/grad
    im = ax[1,1].imshow(tmp,interpolation='nearest',origin='lower',cmap='Spectral',vmin=0,vmax=5)
    ax[1,1].set_aspect('equal')
    ax[1,1].set_title('stddev / local slope')
    pl.colorbar(im, ax=ax[1,1])

    return