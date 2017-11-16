import glob, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


class aperturePhotometry():
    '''
    Do aperture photometry of a star in jpeg images located at ``dirpath``. 
    
    The filenames of the data images must be 'science_*.jpg' for the data, 
    'flats_*.jpg' for the flats and 'bias_*.jpg' for the biases, where * is the
    time step.
        
    Parameters
    ==========
    dirpath : string
        Relative or full path to the directory in which the data is stored. The
        output will be saved here in a subdirectory named ``out``.
    bias : boolean
        Subtract master bias if ``True``. Default is ``False``.
    flat : boolean
        Divide by the normalized master flat if ``True``. Default is ``True``.
    stampsize : int
        Half the size in pixels in each dimension of the stamps. Default is 30.
    radius : int
        Radius of the aperture in pixels. Default is 15.
    bkginner : int
        Inner radius in pixels of the annulus used to calculate the background.
    bkgouter : int
        Outer radius in pixels of the annulus used to calculate the background.
    saveplots : boolean
        Save plots to files if ``True``. Default is ``True``.
        
    Usage
    =====
    Call:
        ``myphot = aperturePhotometry(<path to data directory>)``
    Access flux:
        ``myphot.flux``
    Acess raw flux:
        ``myphot.rawflux``
    Acess background level:
        ``myphot.bkglevel``
    Acess time:
        ``myphot.time``
    
    Methods
    =======
    :func:`~aperturePhotometry.aperturePhotometry.makestamp`
    :func:`~aperturePhotometry.aperturePhotometry.photometry`
    :func:`~aperturePhotometry.aperturePhotometry.findtargets`
    :func:`~aperturePhotometry.aperturePhotometry.aperture`
    :func:`~aperturePhotometry.aperturePhotometry.plothistogram`
    :func:`~aperturePhotometry.aperturePhotometry.plotaperture`
    :func:`~aperturePhotometry.aperturePhotometry.markpixels`
    :func:`~aperturePhotometry.aperturePhotometry.plottimeseries_one`
    :func:`~aperturePhotometry.aperturePhotometry.plottimeseries`
    :func:`~aperturePhotometry.aperturePhotometry.loaddata`
    :func:`~aperturePhotometry.aperturePhotometry.getMasterflat`
    :func:`~aperturePhotometry.aperturePhotometry.getBias`
    
    
    
    Jonas Svenstrup Hansen, november 2017, jonas.svenstrup@gmail.com
    '''
    def __init__(self, dirpath, bias = False, flat = True, stampsize = 30,
                 radius = 15, bkginner = 20, bkgouter = 25,
                 saveplots = True):
        self.dirpath = dirpath
        self.stampsize = stampsize
        self.radius = radius
        self.bkginner = bkginner
        self.bkgouter = bkgouter
        
        # Create directory for output at dirpath:
        self.outdirpath = os.path.join(self.dirpath,'out')
        if not os.path.exists(self.outdirpath):
            os.makedirs(self.outdirpath)
            
        # Load data:
        self.data = self.loaddata('science')
        # Subtract bias is bias parameter is true:
        self.masterbias, self.biaslevel, self.biasstd = self.getBias()
        if bias:
            self.data -= self.masterbias    
            
        # Divide by normalized flat if flat parameter is true:
        self.masterflat = self.getMasterflat()
        if flat:
            self.data /= self.masterflat
        
        # Preallocate:
        self.flux = np.zeros(self.data.shape[0])
        self.rawflux = np.zeros_like(self.flux)
        self.bkglevel = np.zeros_like(self.flux)
        self.time = np.arange(np.shape(self.data)[0])
        
        # Loop over each image:
        for i in range(len(self.time)):
            # Find target(s):
            coord = self.findtargets(self.data[i,:,:])
            # TODO: extend to multiple targets
#            nstars = np.shape(coord)[0]
            nstars = 1
            if nstars is 1:
                # Make stamp:
                stamp, stampcoord, stamprow0, stamprow1, stampcol0, stampcol1 = \
                    self.makestamp(i, coord, stampsize)
                # Do photometry:
                self.rawflux[i], self.bkglevel[i], maskfilter, bkgfilter = \
                    self.photometry(stamp, stampcoord)
                # Plot aperture:
                self.plotaperture(stamp, maskfilter, bkgfilter, i,
                                  stamprow0, stamprow1, stampcol0, stampcol1,
                                  save = saveplots)
            else:
                raise NotImplementedError('Multiple stars in frame not implemented')
        self.flux = self.rawflux - self.bkglevel
        
        # Plot timeseries:
        self.plottimeseries_one(self.flux, 'flux', save = saveplots)
        self.plottimeseries_one(self.rawflux, 'rawflux', save = saveplots)
        self.plottimeseries_one(self.bkglevel, 'bkglevel', save = saveplots)
        self.plottimeseries(self.flux, self.bkglevel, save = saveplots)
        
        
    def makestamp(self, i, coord, stampsize):
        '''
        Create a stamp from data at the time ``i`` around the target at 
        ``coord`` with a size in each direction from that coordinate of
        ``stampsize``.
        
        Parameters
        ==========
        i : int
            Time step. Used to acces ``data`` as ``data[i,.,.]``.
        coord : numpy array (1D)
            Pixel coordinates of the target on the full frame image.
        stampsize : int
            Half the size in each dimension of the stamp.
        
        Returns
        =======
        stamp : numpy array (2D)
            Rectangular cutout of the data with ``stampsize`` number of pixels
            from the target to each edge.
        stampcoord : numpy array (1D)
            Pixel coordinates of the target on the stamp.
        stamprow0 : int
            Row pixel position of the smallest row index of the stamp in the 
            data.
        stamprow1 : int
            Row pixel position of the largest row index of the stamp in the 
            data.
        stampcol0 : int
            Column pixel position of the smallest column index of the stamp in
            the data.
        stampcol1 : int
            Column pixel position of the largest column index of the stamp in
            the data.
        '''
        stamprow0 = coord[0] - stampsize
        stamprow1 = coord[0] + stampsize
        stampcol0 = coord[1] - stampsize
        stampcol1 = coord[1] + stampsize
        stamp = self.data[i,stamprow0:stamprow1,stampcol0:stampcol1]
        stampcoord = [np.shape(stamp)[0]/2, np.shape(stamp)[1]/2]
        return stamp, stampcoord, stamprow0, stamprow1, stampcol0, stampcol1
        
    
    def photometry(self, stamp, stampcoord, bkgmethod = 'median'):
        '''
        Calculate the raw flux and background level of the target with
        coordinates ``stampcoord`` in the image ``stamp`` using aperture 
        photometry.
        
        Parameters
        ==========
        stamp : numpy array (2D)
            Rectangular cutout of the data just around the target.
        stampcoord : numpy array (1D)
            Pixel coordinates on the stamp of the target.
        bkgmethod : string
            Method with which to calculate the background. Default is 
            ``'median'``. Other available methods are: ``'mean'``.
        '''
        maskfilter = self.aperture(stamp, stampcoord, self.radius)
        rawflux = np.nansum(stamp[maskfilter])
        
        bkgfilter = self.aperture(stamp, stampcoord, self.bkginner) ^ \
                    self.aperture(stamp, stampcoord, self.bkgouter)
        if bkgmethod is 'median':
            bkglevel = np.median(stamp[bkgfilter])
        elif bkgmethod is 'mean':
            bkglevel = np.mean(stamp[bkgfilter])
        else:
            raise NotImplementedError('Only median and mean implemented.')
            
        return rawflux, bkglevel, maskfilter, bkgfilter
    
    
    def findtargets(self, datai, maxiter = 10):
        '''
        Find target in the data array ``datai``. The current implementation just
        finds the coordinates of the pixel with the maximum value and only
        works when there is one star in ``datai``.
        
        Parameters
        ==========
        datai : numpy array (2D)
            Data image in which to find targets.
        maxiter : int
            (not implemented) Maximum number of stars to find.
        '''
        return np.asarray(np.unravel_index(np.argmax(datai), np.shape(datai)))
#        # FIX: allow multiple stars:
#        coord = []
#        morestars = True
#        ii = 0
#        while morestars:
#            # Get (row,col) of pixel with maximum value:
#            coord.append(np.unravel_index(np.argmax(datai), np.shape(datai)))
#            # Set values in radius around star to nan:
#            datai[self.aperture(datai, coord[end-1:end], self.bkginner)] = math.nan
#            # Find only maxiter number of star coordinate tuples:
#            ii += 1
#            if ii > maxiter:
#                morestars = False
#            
#        return np.asarray(coord).resize([len(coord)/2,2])
    

    def aperture(self, stamp, coord, radius):
        ''' 
        Make a circular aperture filter the size of stamp around the coordinates
        ``coord`` with given ``radius``.
        
        Parameters
        ==========
        stamp : numpy array (2D)
            Rectangular cutout of the data with the target at the center.
        coord : numpy array (1D)
            Coordinates of the target on the stamp.
        radius : int
            Radius of the aperture to apply.
            
        Returns
        =======
        out : boolean array (2D)
            Boolean two-dimensional array like stamp with True if within 
            ``radius `` of ``coord``.
        '''
        row = coord[0]
        col = coord[1]
        collist, rowlist = np.meshgrid(np.arange(stamp.shape[1]), 
                                       np.arange(stamp.shape[0]))
        out = ((rowlist - row)/radius)**2 + ((collist - col)/radius)**2 < 1
        assert(out.any()) # check that some are true
        return out
        
    
    def plothistogram(self, data, name, nbins=500, save=True):
        '''
        Plot a histogram of the given data using ``matplotlib.pyplot.hist``.
        
        Parameters
        ==========
        data : numpy array (1D)
            One dimensional array of the data to plot a histogram of.
        name : str
            String name of the value. Used in the filename.
        nbins : int
            Number of bins in the histogram. Default is 500.
        save : boolean
            Determine whether to save the plot in ``self.outdirpath``. Default
            is ``True``.
        '''
        fig = plt.figure()
        plt.hist(np.array.flatten(data),nbins)
        plt.xlabel('Bin')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(self.outdirpath, name + '_hist.pdf'))
        plt.close(fig)
    
    
    def plotaperture(self, stamp, maskfilter, bkgfilter, i,
                     stamprow0, stamprow1, stampcol0, stampcol1,
                     maskcolor='black', bkgcolor='red', save=True):
        '''
        Plot a stamp with the aperture and background marked using 
        ``self.markpixels``.
        
        Parameters
        ==========
        stamp : numpy array (2D)
            Two dimensional numpy array of the stamp to plot.
        maskfilter : boolean array (2D)
            Two dimensional boolean array of the same size as stamp with
            ``True`` for pixels in the mask and ``False`` for every other pixel.
        bkgfilter : boolean array (2D)
            Two dimensional boolean array of the same size as stamp with
            ``True`` for pixels in the background and ``False`` for every other 
            pixel.
        i : int
            Time step.
        stamprow0 : int
            Row pixel position of the smallest row index of the stamp in the 
            data.
        stamprow1: int
            Row pixel position of the largest row index of the stamp in the 
            data.
        stampcol0: int
            Column pixel position of the smallest column index of the stamp in
            the data.
        stampcol1: int
            Column pixel position of the largest column index of the stamp in
            the data.
        maskcolor : str
            Color of the edge markings on the pixels in the mask.
        bkgcolor : str
            Color of the edge markings on the pixels in the background.
        save : boolean
            Determine whether to save the plot in ``self.outdirpath``. Default
            is ``True``.
        '''
        fig, ax = plt.subplots()
        plt.imshow(stamp)
        self.markpixels(maskfilter, 'black')
        self.markpixels(bkgfilter, 'red')
        ax.set_xticklabels(np.arange(stampcol0,stampcol1))
        ax.set_yticklabels(np.arange(stamprow0,stamprow1))
        plt.savefig(os.path.join(self.outdirpath, 'aperture' + np.str(i) + '.png'))
        plt.close(fig)
        
        
    def markpixels(self, mask, color='red'):
        '''
        Mark the pixels in a plot by adding square patches along the edges of 
        the pixels in ``mask``.
        
        Parameters
        ==========
        mask : boolean array (2D)
            Two dimensional boolean array of the pixels to mark.
        color : string
            String that determines the color of the marking. Default is 
            ``'red'``.
        '''
        maskidxs = np.column_stack(np.where(mask.transpose()))
        maskidxs = maskidxs - 0.5 # move from center to lower left of pixel
        for maskidx in maskidxs:
            path = patches.Rectangle(maskidx, 1, 1,
                                     facecolor='none', edgecolor=color, linewidth=1,
                                     joinstyle='round')
            plt.gca().add_patch(path)
        
        
    def plottimeseries_one(self, value, name, time=None, save=True):
        '''
        Plot a time series of the given data.
        
        Parameters
        ==========
        value : numpy array (1D)
            Value array of one dimension to plot.
        name : str
            String name of the value. Used in the filename, label in the
            legend and on the y axis.
        time : numpy array (1D)
            Time array of one dimension to plot. Default is ``None`` which 
            creates an array from 0 to ``len(value)`` in intervals of 1.
        '''
        if time is None:
            time = range(len(value))
        fig = plt.figure()
        plt.plot(value,label=name)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(name)
        plt.savefig(os.path.join(self.outdirpath, name + '_timeseries.pdf'))
        plt.close(fig)
    
    
    def plottimeseries(self, flux, bkglevel, time=None, save=True):
        '''
        Plot a time series of the given data and background level with the data
        values on one y axis and the background level on the other.
        
        Parameters
        ==========
        flux : numpy array (1D)
            Flux array of one dimension.
        bkglevel : numpy array (1D)
            Background level array of one dimension.
        time : numpy array (1D)
            Time array of one dimension. Default is ``None`` which creates an
            array from 0 to ``len(flux)`` in intervals of 1.
        save : boolean
            Determine whether to save figures in the out directory 
            ``self.outdirpath``. Default is ``True``.
        '''
        if time is None:
            time = range(len(flux))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(flux,color='black',label='flux')
        ax2.plot(bkglevel,color='red',label='bkglevel')
        
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Flux')
        ax2.set_ylabel('Background level')
        
        plt.savefig(os.path.join(self.outdirpath, 'timeseries.pdf'))
        plt.close(fig)
    
    
    def loaddata(self, datatype):
        '''
        Load jpeg data of the given type from the directory ``self.dirpath``.
        
        Parameters
        ==========
        datatype : str
            Data type to look for in the directory ``self.dirpath``. Can be 
            ``'science'``, ``'flats'`` or ``'bias'``.
        
        Returns
        =======
        data : numpy array
            Array containing the data in ``file`` with datatype ``'float32'``.
        '''
        data = []
        for file in glob.glob(self.dirpath + datatype + '_*.jpg'):
            with Image.open(file) as im:
                data.append(np.asarray(im, dtype='float32'))
        return np.asarray(data)
    
    
    def getMasterflat(self):
        '''
        Load flat files and calculate the normalized master flat.
        
        Returns
        =======
        masterflat : numpy array
            Normalized master flat shaped like a data frame.
        '''
        flats = self.loaddata('flats')
        #flats -= masterbias # error: the flats are of too low flux
        for flat in range(np.shape(flats)[0]):
            flatmean = np.mean(flats[flat,:,:])
            flats[flat,:,:] /= flatmean
        masterflat = np.mean(flats, 0)
        return masterflat
    
    
    def getBias(self):
        '''
        Load bias files and calculate the master bias.
        
        Returns
        =======
        masterbias : numpy array
            Master bias shaped like a data frame.
        biaslevel : float
            Mean of the master bias.
        biasstd : float
            Standard deviation of the values in the master bias.
        '''
        biases = self.loaddata('bias')
        masterbias = np.mean(biases, 0)
        biasstd = np.std(masterbias)
        biaslevel = np.mean(masterbias)
        return masterbias, biaslevel, biasstd
        