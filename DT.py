import numpy as np
import tensorflow as tf

class DT:
    def __init__(self, scattering_model='born', im_size=200, xy_upsamp=1, z_upsamp=2, xy_fov_upsamp=1, z_fov_upsamp=1,
                 sample_thickness_MS=100, sample_pix_MS=100, use_spatial_patching=False, im_size_full=None):
        # scattering_model can be 'born' or 'multislice';
        # im_size is the side length of the raw data;
        # xy_upsamp is a float that specifies how many times larger the xy recon should be than default (interfaces into
        # kxy_max); likewise for z_upsamp;
        # xy_fov_upsamp and z_fov_upsamp also increases the recon size, but not by changing k_max, but rather increasing
        # the sampling density in k-space;
        # sample_thickness_MS and sample_pix_MS only pertain to when the scattering model is multslice; the former is
        # the thickness in um and the latter is the thickness in pixels;
        # if use_spatial_patching, then set the im_size_full to the full data side length;
        # feel free to modify the other hyperparameters below after running this constructor, though be warned that
        # some of the hyperparameters defined here depend on others;

        self.scattering_model = scattering_model
        self.use_spatial_patching = use_spatial_patching  # using random crops; the crop size is self.xy_cap_n
        self.n_back = 1.515  # immersion index
        self.xy_cap_n = im_size  # number of pixels one one side
        self.lambdas = np.array([.632], dtype=np.float32)  # wavelength in um
        self.k_vacuum = 1 / self.lambdas  # vacuum wavenumber; spatial frequency, not angular frequency
        self.k_illum = self.k_vacuum * self.n_back  # medium wavenumber
        self.NA = .4  # imaging numerical aperture;
        self.th_aper = np.arcsin(self.NA)  # maximum collection half angle;
        self.magnification = 30
        self.pixel_pitch = 4.54  # um
        self.dxy_sample = self.pixel_pitch / self.magnification  # spacing at sample for a camera image in um
        self.xy_LED = 31  # number LEDs on one side for the square LED array
        self.LED_pitch = 4  # spacing between LEDs in mm
        self.LED_dist2samp = 144  # distance of LED board from sample in mm
        self.max_angle_illum = np.arctan(self.xy_LED // 2 * self.LED_pitch / self.LED_dist2samp)
        self.max_angle_illum_diag = np.arctan(self.xy_LED // 2 * np.sqrt(2) * self.LED_pitch / self.LED_dist2samp)
        self.NA_illum = np.sin(self.max_angle_illum)  # illumination NA
        self.NA_illum_diag = np.sin(self.max_angle_illum_diag)  # max illumination angle (along diagonal of square)
        self.xy_upsamp = xy_upsamp
        self.z_upsamp = z_upsamp

        if self.scattering_model == 'multislice':
            # the recon size in xy will be same as that of the data (unless using spatial patching)
            self.dxy = self.dxy_sample * self.xy_upsamp
            self.side_kxy = self.xy_cap_n
            self.dz = sample_thickness_MS / sample_pix_MS
            self.side_kz = sample_pix_MS
            self.sample_thickness = sample_thickness_MS
            self.apod_frac = .4  # for apodizing the propogated fields
            self.optimize_focal_shift = False  # allow the focus for multislice to change?
        elif self.scattering_model == 'born':
            self.kxy_max = self.k_illum.max() * (  # max resolvable spatial freq in xy
                    self.NA + self.NA_illum) * self.xy_upsamp
            self.kz_max = self.k_illum.max() * (  # max resolvable spatial freq in z
                    1 - np.sqrt(1 - np.maximum(self.NA, self.NA_illum_diag) ** 2)) * z_upsamp
            self.dxy = 1 / 2 / self.kxy_max  # xy spacing in reconstruction
            self.dz = 1 / 2 / self.kz_max # z spacing in reconstruction
            # dimensions of reconstruction; let sampling dictate discretization:
            self.side_kxy = np.int32(np.ceil(self.xy_cap_n * self.dxy_sample / self.dxy * xy_fov_upsamp))
            self.side_kz = np.int32(np.ceil(self.side_kxy * self.kz_max / self.kxy_max * z_fov_upsamp))
        else:
            raise Exception('invalid scattering model:' + self.scattering_model)

        self.xy_max = self.dxy * self.side_kxy  # physical lateral dimensions in um
        self.recon_shape = (self.side_kxy, self.side_kxy, self.side_kz)
        if self.use_spatial_patching:
            assert im_size_full is not None
            self.xy_full_n = im_size_full
            xy_full = np.int32(np.ceil(self.side_kxy * self.xy_full_n / self.xy_cap_n))
            self.recon_shape_full = (xy_full, xy_full, self.side_kz)

        # theta and phi coordinates of the aperture locations:
        self.theta_apers = np.array([0], dtype=np.float32)
        self.phi_apers = np.array([0], dtype=np.float32)
        self.num_apers = len(self.theta_apers)

        # regularization:
        self.TV_reg_coeff = 0  # coefficient of the TV regularization term
        self.positivity_reg_coeff = 0
        self.negativity_reg_coeff = 0
        self.use_deep_image_prior = False
        self.numfilters_list = [16, 32, 64, 128, 128]  # number of filters in the upsample/downsample blocks
        self.numskipfilters_list = [0, 0, 0, 0, 0]  # must be same length as ^
        self.DIP_lr = tf.placeholder_with_default(.001, shape=(), name='DIP_lr')  # feed_dict to modify during training;
        self.DIP_make_pow2 = False  # makes dimensions a power of 2; make true if using skip connections
        self.normalizing_layer = lambda input_layer: tf.layers.batch_normalization(input_layer, training=True)
        # ^what normalizing layer to use in the DIP?
        self.linear_DIP_output = True  # should the last layer have a linear activation or leaky_relu?
        self.DIP_upsample_method = 'nearest'  # 'nearest' or 'bilinear'
        self.DIP_output_scale = 1./400  # to scale the output of the DIP network

        # other settings:
        self.batch_size = 32  # number of angles to use
        self.data_ignore = None  # None, or a boolean vector specifying whether to ignore each LED
        self.DT_recon_r_initialize = None  # initializer for DT
        self.optimize_k_directly = False  # whether you're setting the tf.Variable to k-space or real-space
        self.pupil_function = False  # whether to optimize for the pupil function
        self.zero_out_background_if_outside_aper = False
        self.coordinate_offset = np.array([0, 0, 0])  # if you get low freq modulation artifacts, try a pixel shift;
        # (e.g., -1, -.5, 0, .5, 1)
        self.force_pass_thru_DC = True  # sometimes the bowls won't pass through DC; force them to?
        self.DC_pixel = np.ceil(np.array((self.side_kxy / 2, self.side_kxy / 2, self.side_kz / 2))).astype(np.int32)
        # ^if forcing the bowls to go through DC, which pixel is DC?
        self.FDT_calibration_factor = 20 / 3  # calibration factor for FDT to get the RI right
        self.train_DC = True  # whether to optimize the background illumination amplitude
        self.focus_init = 0  # for multislice, the inital focal shift in um
        self.kspace_lr_scale = 200.  # to scale the learning rate for k-space optimization

    def generate_cap(self):
        # generate Ewald spherical cap;
        self.generate_k_coordinates()
        # project onto sphere:
        self.kz_cap = tf.sqrt(tf.maximum(self.k_cap ** 2 - self.kx_cap ** 2 - self.ky_cap ** 2, 0))
        self.xyz_cap = tf.stack([self.kx_cap, self.ky_cap, self.kz_cap], 0)
        self.kz_cap = tf.reshape(self.kz_cap, [-1])  # for the prefactor in the Fourier diffraction theorem
        # ... for now, assume only one illumination color;
        self.kz_cap += 1 - tf.reshape(self.aperture_mask[:, :, 0], [-1])  # to avoid divide by 0 outside of aperture
        self.kz_cap = tf.to_float(self.kz_cap)

    def generate_k_coordinates(self):
        # generate the aperture mask for isolating circle in Fourier space:
        # (different radius for different k_illum)
        kxy_cap = np.arange(self.xy_cap_n, dtype=np.float32)
        kxy_cap -= np.mean(kxy_cap)  # center
        kxy_cap *= 1 / self.dxy_sample / self.xy_cap_n  # multiply by spacing in um^-1
        # shape: xy_cap_n by xy_cap_n by len(k_illum):
        self.kx_cap, self.ky_cap, self.k_cap = tf.meshgrid(kxy_cap, kxy_cap, self.k_illum)
        self.aperture_mask = tf.to_float(tf.less(self.kx_cap ** 2 + self.ky_cap ** 2, (self.k_cap * self.NA) ** 2))
        self.evanescent_mask = tf.to_float(self.kx_cap ** 2 + self.ky_cap ** 2 < self.k_cap ** 2)

    def generate_apertures(self):
        # rotate the sampled spherical caps by the angle corresponding to the angular location of the aperture(s);
        roty = np.stack([np.cos(self.phi_apers), np.zeros(self.num_apers), np.sin(self.phi_apers),
                         np.zeros(self.num_apers), np.ones(self.num_apers), np.zeros(self.num_apers),
                         -np.sin(self.phi_apers), np.zeros(self.num_apers), np.cos(self.phi_apers)])
        roty = np.reshape(roty, (3, 3, self.num_apers))
        roty = np.transpose(roty, (2, 0, 1))
        rotz = np.stack([np.cos(self.theta_apers), -np.sin(self.theta_apers), np.zeros(self.num_apers),
                         np.sin(self.theta_apers), np.cos(self.theta_apers), np.zeros(self.num_apers),
                         np.zeros(self.num_apers), np.zeros(self.num_apers), np.ones(self.num_apers)])
        rotz = np.reshape(rotz, (3, 3, self.num_apers))
        rotz = np.transpose(rotz, (2, 0, 1))
        self.rot = np.matmul(rotz, roty).astype(np.float32)
        xyz_cap_flat = tf.reshape(self.xyz_cap, (3, -1))  # to allow rotation via matmul
        # shape: num apertures by 3 by number of points in cap:
        self.xyz_caps = tf.matmul(self.rot, tf.tile(xyz_cap_flat[None], (self.num_apers, 1, 1)))
        # reshape back:
        self.xyz_caps = tf.reshape(self.xyz_caps, (self.num_apers,  # apertures
                                                   3,  # xyz
                                                   self.xy_cap_n,  # camera dim
                                                   self.xy_cap_n,  # camera dim
                                                   len(self.k_illum)))  # color

    def generate_LED_positions_flat_array(self):
        # xyz LED positions:
        xy_LED_pos = np.arange(0, self.xy_LED, dtype=np.float32) * self.LED_pitch
        xy_LED_pos -= np.mean(xy_LED_pos)
        y_LED, x_LED = np.meshgrid(xy_LED_pos, xy_LED_pos)
        x_LED = x_LED.flatten()
        y_LED = y_LED.flatten()
        z_LED = self.LED_dist2samp
        # spherical (without r), then back to cartesian
        # for some reason tf's acos returns nan for -1 and 1:
        phi_LED = tf.acos(tf.clip_by_value(z_LED / tf.sqrt(x_LED ** 2 + y_LED ** 2 + z_LED ** 2), -.9999999, .9999999))
        theta_LED = tf.atan2(y_LED, x_LED)
        self.xyz_LED = tf.stack([tf.sin(phi_LED) * tf.cos(theta_LED),
                                 tf.sin(phi_LED) * tf.sin(theta_LED),
                                 tf.cos(phi_LED)])  # shape: 3 by number of LEDs

    def subtract_illumination(self):
        # given that xyz_caps are generated for every aperture, this function subtracts out the illumination vector so
        # that the spherical caps are moved to the correct location in k-space; use these coordinates to sample the
        # object k-space;

        # subtract out input illumination (xyz_LED is direction of unit mag; multiply by k):
        # shape: 3, num illum (colors), number of LEDs
        self.k_illum_vectors = self.xyz_LED_batch[:, None, :] * self.k_illum[None, :, None]
        # shape: num apertures, 3, camx, camy, num illum, number of LEDs:
        self.xyz_caps = (self.xyz_caps[:, :, :, :, :, None] - self.k_illum_vectors[None, :, None, None, :, :])

        # move the xyz dim to the end (for gather_nd/scatter_nd):
        # new shape: num aper, num LEDs, num illum, camx, camy, 3
        self.xyz_caps = tf.transpose(self.xyz_caps, (0, 5, 4, 2, 3, 1))
        # new shape: num illum, camx, camy
        self.aperture_mask = tf.transpose(self.aperture_mask, (2, 0, 1))

        # flatten aperture and LED dims:
        self.xyz_caps = tf.reshape(self.xyz_caps, (-1, self.points_per_cap, 3))
        self.xyz_caps_continuous = tf.to_float(self.xyz_caps)  # save this for diagnostics

        # discretize coordinates (from -k_max:k_max to 0:side_k-1):
        # first, create matrix with all the side_ks and k_maxes:
        self.side_k = np.array([self.side_kxy, self.side_kxy, self.side_kz], dtype=np.float32)
        self.k_max = np.array([self.kxy_max, self.kxy_max, self.kz_max], dtype=np.float32)

        self.xyz_caps = (self.xyz_caps / self.k_max / 2 + .5) * self.side_k - self.coordinate_offset[None, None, :]

        if self.force_pass_thru_DC:
            # find the closest pixel to DC and shift the whole bowl by the distance;
            # for most bowls, the closet pixel is already at DC so if the distance >1 pixel, something is wrong ...
            # xyz_caps is num_caps by _ by 3;

            # compute based on the non-rounded values -- this is to reduce ambiguity if multiple pixels are close;
            self.diff = self.xyz_caps - tf.to_float(self.DC_pixel[None, None])  # difference vectors
            self.dists = self.diff[:, :, 0] ** 2 + self.diff[:, :, 1] ** 2 + self.diff[:, :, 2] ** 2
            self.min_dist_to_DC = tf.reduce_min(self.dists, axis=1)
            self.closest_points_to_DC = tf.argmin(self.dists, axis=1, output_type=tf.int32)
            self.DC_adjust = tf.gather_nd(self.diff,
                                          tf.stack([tf.range(tf.shape(self.batch_inds)[0]),  # NOT self.batch_inds!
                                                    self.closest_points_to_DC], axis=1))  # num_caps by 3
            self.xyz_caps -= self.DC_adjust[:, None, :]

        self.xyz_caps = tf.round(self.xyz_caps)
        self.xyz_caps = tf.to_int32(self.xyz_caps)
        self.xyz_caps = tf.clip_by_value(self.xyz_caps, tf.zeros_like(self.xyz_caps), self.side_k[None, None] - 1)
        self.min_dist_to_DC_after_adjust = tf.reduce_min(  # for diagnostic purposes
            tf.reduce_sum((self.xyz_caps - self.DC_pixel[None, None]) ** 2, axis=2), axis=1)
        self.generate_background()

    def generate_background(self):
        # illumination background;

        kx_illum, ky_illum, kz_illum = tf.split(self.k_illum_vectors, 3, axis=0)  # num illum, num LEDs
        kx_illum = kx_illum[0]  # remove the split dimension
        ky_illum = ky_illum[0]
        kz_illum = kz_illum[0]

        # create mask that zeros out illuminations that miss the aperture:
        # reshape to _ by 1
        self.miss_aper_mask = tf.to_float(
            tf.less(kx_illum ** 2 + ky_illum ** 2, (self.k_illum * self.NA) ** 2))[0, :, None]

        # if shifting bowls to force passage thru DC, modify illumination kxy:
        if self.force_pass_thru_DC:
            # kx_illum is num illum x num LEDs
            # DC_adjust is num_LEDs x 3
            kx_illum += (self.DC_adjust[None, :, 0]) * self.k_max[0] * 2 / self.side_k[0]
            ky_illum += (self.DC_adjust[None, :, 1]) * self.k_max[1] * 2 / self.side_k[1]
            kz_illum += (self.DC_adjust[None, :, 2]) * self.k_max[2] * 2 / self.side_k[2]

            # renormalize magnitude to k_illum:
            k_mag = tf.sqrt(kx_illum ** 2 + ky_illum ** 2 + kz_illum ** 2)
            kx_illum *= self.k_illum / k_mag
            ky_illum *= self.k_illum / k_mag

        # generate 2D phase ramp, for 0-reference fft:
        xy_samp = np.arange(self.xy_cap_n, dtype=np.float32)
        xy_samp -= np.ceil(self.xy_cap_n / 2)  # center
        xy_samp *= self.dxy_sample  # image coordinates
        x_samp, y_samp = tf.meshgrid(xy_samp, xy_samp)
        x_samp = tf.reshape(x_samp, [-1])
        y_samp = tf.reshape(y_samp, [-1])
        # shape: num illum, num LEDs, camx*camy:
        self.k_fft_shift = tf.exp(1j * 2 * np.pi * tf.to_complex64(x_samp[None, None, :] * kx_illum[:, :, None] +
                                                                   y_samp[None, None, :] * ky_illum[:, :, None]))
        # squeeze for now, assuming one illumination for now:
        # this is actually already batched because derived from xyz_LED_batch:
        self.k_fft_shift = tf.squeeze(self.k_fft_shift)

    def format_DT_data(self, stack, DC=None):
        # expects an input stack of shape: num aper, num LEDs, num illum, camx, camy;
        # do not take sqrt of the data -- that is done here;

        s = stack.shape
        assert self.num_apers == s[0]
        if not self.use_spatial_patching:
            # if using spatial patching, then s[3]=s[4]>xy_cap_n
            assert s[3] == s[4] == self.xy_cap_n
        else:
            assert s[3] == s[4] == self.xy_full_n

        self.num_caps = s[0] * s[1]  # number of spherical caps (aper*LED)
        self.points_per_cap = s[2] * self.xy_cap_n ** 2  # for every color

        self.data_stack = np.reshape(stack, (self.num_caps, s[3] ** 2))
        self.data_stack = np.sqrt(self.data_stack)  # so that we don't have to do this for each new batch

        # DC due to unscattered light, potentially different for every angle:
        if DC is None:
            # initialize from data
            DC = np.median(self.data_stack, 1)
        self.DC = tf.Variable(DC, dtype=np.float32, name='DC')
        self.illumination_phase = tf.Variable(tf.zeros(self.num_caps, dtype=tf.float32),
                                              name='illumination_phase', trainable=False)

        self.generate_LED_positions_flat_array()

        if self.use_spatial_patching:
            # this implementation doesn't finish all the LEDs in one spatial crop before moving to another;

            # upper left hand corner of the crop to be made:
            self.spatial_batch_inds = tf.random_uniform(shape=(2, 1), minval=0,
                                                        maxval=self.xy_full_n - self.xy_cap_n, dtype=tf.int32)
            # batch along LED dimension:
            self.dataset = (tf.data.Dataset.range(self.num_caps)
                            .shuffle(self.num_caps)
                            .batch(self.batch_size)
                            .repeat(None)
                            .make_one_shot_iterator())
            self.batch_inds = self.dataset.get_next()
            # reshape so that we can crop:
            self.data_stack = self.data_stack.reshape(self.num_caps, self.xy_full_n, self.xy_full_n)
        else:
            # generate dataset for batching:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.data_stack, tf.range(self.num_caps)))
            if self.batch_size != self.num_caps:
                # if all examples are present, don't shuffle
                self.dataset = self.dataset.shuffle(self.num_caps)
            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.repeat(None)  # go forever
            self.batcher = self.dataset.make_one_shot_iterator()
            (self.data_batch, self.batch_inds) = self.batcher.get_next()

        if self.data_ignore is not None:
            keep_inds = tf.gather(~self.data_ignore, self.batch_inds)
            self.batch_inds = tf.boolean_mask(self.batch_inds, keep_inds)
            if not self.use_spatial_patching:
                # data batch is generated using data_inds for spatial patching
                self.data_batch = tf.boolean_mask(self.data_batch, keep_inds)

        self.DC_batch = tf.gather(self.DC, self.batch_inds)
        self.DC_batch = tf.to_complex64(self.DC_batch[:, None])
        self.illumination_phase_batch = tf.gather(self.illumination_phase, self.batch_inds)
        self.xyz_LED_batch = tf.transpose(  # transpose because first dim is 3 for xyz
            tf.gather(tf.transpose(self.xyz_LED), self.batch_inds))

    def spatial_patching(self):
        assert self.scattering_model == 'multislice'

        ULcorner = self.spatial_batch_inds[:, 0]
        begin = tf.concat([[0], ULcorner], axis=0)  # data_intensity is num_caps by x by y
        crop = tf.slice(self.data_stack, begin=begin, size=(-1, self.xy_cap_n, self.xy_cap_n))
        self.data_batch = tf.gather(crop, self.batch_inds)  # batch along LED dim
        self.data_batch = tf.reshape(self.data_batch, [-1, self.xy_cap_n ** 2])

        if self.use_deep_image_prior:
            # needs to be handled in the deep_image_prior function because that is run before this function
            pass
        else:
            begin = tf.concat([ULcorner, [0]], axis=0)  # DT_recon is num_caps by x by y by z
            self.DT_recon_sbatch = tf.slice(self.DT_recon, begin=begin,
                                             size=(self.xy_cap_n, self.xy_cap_n, -1))

    def reconstruct_with_multislice(self):
        # only two parameterization options: direct index recon, or DIP index recon;

        assert self.force_pass_thru_DC is False  # bowls are not generated, so this can't be done
        assert self.optimize_k_directly is False  # we are not using k-spheres

        self.k_illum_vectors = self.xyz_LED_batch[:, None, :] * self.k_illum[None, :, None]
        self.generate_background()  # generates the variables needed for the background illumination
        self.k_fft_shift_batch = tf.conj(self.k_fft_shift)

        self.initialize_space_space_domain()
        self.RI = self.DT_recon + self.n_back  # no reference to scattering potential

        if self.use_spatial_patching:
            self.spatial_patching()
            if self.use_deep_image_prior:
                # DT recon is already generated from the spatially cropped input to DIP
                DT_recon = self.DT_recon
            else:
                DT_recon = self.DT_recon_sbatch
        else:
            DT_recon = self.DT_recon

        # fresnel propagation kernel:
        # fix the squeezing in the future if using more than one color
        k0 = np.squeeze(self.k_vacuum)
        kn = np.squeeze(self.k_illum)
        self.generate_k_coordinates()
        kx = tf.to_complex64(tf.squeeze(self.kx_cap))
        ky = tf.to_complex64(tf.squeeze(self.ky_cap))
        self.k_2 = kx ** 2 + ky ** 2
        self.F = tf.exp(-1j * 2 * np.pi * self.k_2 * self.dz / (kn + tf.sqrt(kn ** 2 - self.k_2)))
        self.F *= tf.squeeze(
            tf.to_complex64(self.evanescent_mask))  # technically not needed, but due to numerical instabilities...
        self.F = self.tf_fftshift2(self.F)
        self.F = tf.to_complex64(self.F, name='fresnel_kernel')  # shape: xy_cap_n by xy_cap_n

        # shape: num caps by points per cap:
        self.illumination = self.DC_batch * self.k_fft_shift_batch  # called unscattered in reconstruct_with_born
        self.illumination = tf.reshape(self.illumination,
                                       [-1, self.xy_cap_n, self.xy_cap_n])

        # incorporate additional defocus factor to account for unknown focal position after propagating through sample;
        # 0 corresponds to the center of the sample; distance in um;
        # change the initial position of the beam so that after refocusing, the beam is at the center of the fov;
        self.focus = tf.Variable(self.focus_init, dtype=tf.float32, name='focal_position')

        # create apodizing Gaussian window:
        # use tf.contrib.image.translate rather than recompute for every LED to save time/memory:
        k_max_radius = 1 / 2 / self.dxy_sample  # max possible radius
        # compute shifts (using LED positions):
        x_shift = -(self.focus - self.sample_thickness / 2) * self.xyz_LED[0] / self.xyz_LED[2]
        y_shift = -(self.focus - self.sample_thickness / 2) * self.xyz_LED[1] / self.xyz_LED[2]
        self.xy_shift = tf.stack([x_shift, y_shift], axis=1) / self.dxy  # convert to pixel
        # centered, unshifted gaussian window
        gausswin0 = tf.exp(-tf.to_float(self.k_2) / 2 / (k_max_radius * self.apod_frac) ** 2)
        gausswin = tf.tile(gausswin0[None], (self.num_caps, 1, 1))
        gausswin = tf.contrib.image.translate(gausswin[:, :, :, None], self.xy_shift, 'bilinear')
        self.gausswin = gausswin[:, :, :, 0]  # get rid of color channels
        self.gausswin_batch = tf.gather(self.gausswin, self.batch_inds)

        self.illumination *= tf.to_complex64(self.gausswin_batch)  # gaussian window

        # forward propagation:
        def propagate_1layer(field, t_i):
            # field: the input field;
            # t_i, the 2D object transmittance function at the current (ith) plane, referenced to background index;
            return tf.ifft2d(tf.fft2d(field) * self.F) * t_i

        dN = tf.transpose(DT_recon, [2, 0, 1])  # make z the leading dim
        t = tf.exp(1j * 2 * np.pi * k0 * dN * self.dz)  # transmittance function
        self.propped = tf.scan(propagate_1layer, initializer=self.illumination, elems=t, swap_memory=True)
        self.propped = tf.transpose(self.propped, [1, 2, 3, 0])  # num ill, x, y, z

        self.pupil_phase = tf.Variable(np.zeros((self.xy_cap_n,
                                                 self.xy_cap_n)),
                                       dtype=tf.float32,
                                       name='pupil_phase_function')
        pupil = tf.exp(1j * tf.to_complex64(self.pupil_phase))
        limiting_aperture = tf.squeeze(tf.to_complex64(self.aperture_mask))
        k_2 = self.k_2 * limiting_aperture  # to prevent values far away from origin from being too large
        self.F_to_focus = tf.exp(-1j * 2 * np.pi * k_2 * tf.to_complex64(-self.focus - self.sample_thickness / 2) /
                                 (kn + tf.sqrt(kn ** 2 - k_2)))
        # restrict to the experimental aperture
        self.F_to_focus *= limiting_aperture
        self.F_to_focus *= pupil  # to account for aberrations common to all
        self.F_to_focus = self.tf_fftshift2(self.F_to_focus)
        self.F_to_focus = tf.to_complex64(self.F_to_focus,
                                          name='fresnel_kernel_prop_to_focus')

        self.field = tf.ifft2d(tf.fft2d(self.propped[:, :, :, -1]) * self.F_to_focus[None])
        self.forward_pred = tf.abs(self.field)
        self.forward_pred = tf.reshape(self.forward_pred, [-1, self.xy_cap_n ** 2])

        self.data_batch *= tf.reshape(gausswin0, [-1])[None]  # since prediction is windowed, also window data
        self.generate_train_ops()

    def reconstruct(self):
        if self.scattering_model == 'multislice':
            self.reconstruct_with_multislice()
        else:
            self.reconstruct_with_born()

    def reconstruct_with_born(self):
        # use intensity (no phase) data and try to reconstruct 3D index distribution;

        if self.optimize_k_directly:  # tf variables are k space
            self.initialize_k_space_domain()
        else:  # tf variables are space domain
            self.initialize_space_space_domain()

        # DT_recon is the scattering potiential; then to get RI:
        self.RI = self.V_to_RI(self.DT_recon)

        # generate k-spherical caps:
        self.generate_cap()
        self.generate_apertures()
        self.subtract_illumination()

        # already batched, because derived from xyz_LED_batch:
        self.k_fft_shift_batch = self.k_fft_shift
        self.xyz_caps_batch = self.xyz_caps

        self.pupil_phase = tf.Variable(np.zeros((self.xy_cap_n, self.xy_cap_n)),
                                       dtype=tf.float32, name='pupil_phase_function')
        pupil = tf.exp(1j * tf.to_complex64(self.pupil_phase))

        # error between prediction and data:
        k_space_T = tf.transpose(self.k_space, [1, 0, 2])
        forward_fourier = self.tf_gather_nd3(k_space_T, self.xyz_caps_batch)
        forward_fourier /= tf.complex(0., self.kz_cap[
            None]) * 2  # prefactor; it's 1i*kz/pi, but my kz is not in angular frequency
        forward_fourier = tf.reshape(forward_fourier,  # so we can do ifft
                                     (-1, len(self.k_illum),  # self.batch_size
                                      self.xy_cap_n, self.xy_cap_n))
        # zero out fourier support outside aperture before fftshift:
        forward_fourier *= tf.complex(self.aperture_mask[None], 0.)
        if self.pupil_function:
            forward_fourier *= pupil
        self.forward_pred = self.tf_ifftshift2(tf.ifft2d(self.tf_fftshift2(forward_fourier)))
        # fft phase factor compensation:
        self.forward_pred *= tf.to_complex64(self.dxy ** 2 * self.dz / self.dxy_sample ** 2)
        self.forward_pred = tf.reshape(self.forward_pred,  # reflatten
                                       (-1, self.points_per_cap))  # self.batch_size
        self.field = tf.identity(self.forward_pred)  # to monitor the E field for diagnostic purposes
        unscattered = self.DC_batch * self.k_fft_shift_batch * tf.exp(
            1j * tf.to_complex64(self.illumination_phase_batch[:, None]))

        if self.zero_out_background_if_outside_aper:
            # to zero out background from illumination angles that miss the aperture
            self.miss_aper_mask_batch = tf.to_complex64(self.miss_aper_mask)
            self.forward_pred_field = self.DC_batch * self.forward_pred + unscattered * self.miss_aper_mask_batch
            self.forward_pred = tf.abs(self.forward_pred_field)
        else:
            self.forward_pred_field = self.DC_batch * self.forward_pred + unscattered
            self.forward_pred = tf.abs(self.forward_pred_field)

        self.generate_train_ops()

    def generate_train_ops(self):

        self.MSE = tf.reduce_mean((self.data_batch - self.forward_pred) ** 2)
        self.loss = [self.MSE]

        # only add these additive regularization terms if the coefficient is not essentially 0:
        if self.TV_reg_coeff > 1e-12:
            self.loss.append(self.TV_reg())
        if self.positivity_reg_coeff > 1e-12:
            self.loss.append(self.positivity_reg())
        if self.negativity_reg_coeff > 1e-12:
            self.loss.append(self.negativity_reg())

        loss = tf.reduce_sum(self.loss)
        self.train_op_list = list()

        if self.optimize_k_directly:
            # rescale learning rate depending on input size
            train_op_k = tf.train.AdamOptimizer(
                learning_rate=self.kspace_lr_scale * self.z_upsamp * self.xy_upsamp).minimize(
                loss, var_list=[self.DT_recon_r, self.DT_recon_i])
            self.train_op_list.append(train_op_k)
        else:
            if self.use_deep_image_prior:
                train_op_k = tf.train.AdamOptimizer(learning_rate=self.DIP_lr).minimize(
                    loss, var_list=tf.trainable_variables(scope='deep_image_prior'))
                self.train_op_list.append(train_op_k)
            else:
                if self.scattering_model == 'born':
                    self.lr = .0002
                elif self.scattering_model == 'multislice':
                    self.lr = .00002
                else:
                    raise Exception('invalid scattering model')

                train_op_k = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                    loss, var_list=[self.DT_recon_r, self.DT_recon_i])
                self.train_op_list.append(train_op_k)

        if self.pupil_function:
            train_op_pupil = tf.train.AdamOptimizer(learning_rate=.1).minimize(loss, var_list=[self.pupil_phase])
            self.train_op_list.append(train_op_pupil)

        if self.scattering_model == 'multislice' and self.optimize_focal_shift:
            train_op_focus = tf.train.AdamOptimizer(learning_rate=.1).minimize(loss, var_list=[self.focus])
            self.train_op_list.append(train_op_focus)

        if self.train_DC:
            train_op_DC = tf.train.AdamOptimizer(learning_rate=.1).minimize(loss, var_list=[self.DC])
            self.train_op_list.append(train_op_DC)

        self.train_op = tf.group(*self.train_op_list)
        self.saver = tf.train.Saver()

    def initialize_space_space_domain(self):
        if self.use_spatial_patching:
            recon_shape = self.recon_shape_full
        else:
            recon_shape = self.recon_shape

        if self.use_deep_image_prior:
            with tf.variable_scope('deep_image_prior'):
                self.deep_image_prior()
        else:
            if self.DT_recon_r_initialize is not None:
                self.DT_recon_r = tf.Variable(self.DT_recon_r_initialize, dtype=tf.float32, name='recon_real')
            else:
                self.DT_recon_r = tf.get_variable(shape=recon_shape, dtype=tf.float32,
                                                  initializer=tf.random_uniform_initializer(0, 1e-7), name='recon_real')
            self.DT_recon_i = tf.get_variable(shape=recon_shape, dtype=tf.float32,
                                              initializer=tf.random_uniform_initializer(0, 1e-7), name='recon_imag')
        self.DT_recon = tf.complex(self.DT_recon_r, self.DT_recon_i)
        self.k_space = self.tf_ifftshift3(tf.fft3d(self.tf_fftshift3(self.DT_recon)))

    def initialize_k_space_domain(self):
        if self.use_spatial_patching:
            recon_shape = self.recon_shape_full
        else:
            recon_shape = self.recon_shape

        self.DT_recon_r = tf.get_variable(shape=recon_shape, dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(0, 1e-7), name='recon_real_k')
        self.DT_recon_i = tf.get_variable(shape=recon_shape, dtype=tf.float32,
                                          initializer=tf.random_uniform_initializer(0, 1e-7), name='recon_imag_k')
        self.k_space = tf.complex(self.DT_recon_r, self.DT_recon_i)
        self.DT_recon = self.tf_ifftshift3(tf.ifft3d(self.tf_fftshift3(self.k_space)))

    def TV_reg(self):
        # total variation regularization
        A = self.DT_recon
        d0 = (A[1:, :-1, :-1] - A[:-1, :-1, :-1])
        d1 = (A[:-1, 1:, :-1] - A[:-1, :-1, :-1])
        d2 = (A[:-1, :-1, 1:] - A[:-1, :-1, :-1])
        return self.TV_reg_coeff * tf.reduce_sum(tf.sqrt(tf.abs(d0) ** 2 + tf.abs(d1) ** 2 + tf.abs(d2) ** 2))

    def positivity_reg(self):
        # the real part of the index doesn't drop below the background index
        negative_components = tf.minimum(tf.real(self.RI) - self.n_back, 0)
        return self.positivity_reg_coeff * tf.reduce_sum(negative_components ** 2)

    def negativity_reg(self):
        positive_components = tf.maximum(tf.real(self.RI) - self.n_back, 0)
        return self.negativity_reg_coeff * tf.reduce_sum(positive_components ** 2)

    def deep_image_prior(self):
        def build_model(net_input):
            def downsample_block(net, numfilters=32, kernel_size=3):
                net = tf.layers.conv3d(net, filters=numfilters, kernel_size=kernel_size,
                                       strides=(2, 2, 2), padding='same')
                net = self.normalizing_layer(net)
                net = tf.nn.leaky_relu(net)

                # repeat, but no downsample this time
                net = tf.layers.conv3d(net, filters=numfilters, kernel_size=3, strides=(1, 1, 1), padding='same')
                net = self.normalizing_layer(net)
                net = tf.nn.leaky_relu(net)

                return net

            def upsample_block(net, numfilters=32, kernel_size=3):
                def upsample3D(A, factor):
                    # because tf only has a 2D version ...
                    # A is of shape [1, x, y, z, channels];
                    # upsample x and y first, then z;

                    if self.DIP_upsample_method == 'nearest':
                        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
                    elif self.DIP_upsample_method == 'bilinear':
                        method = tf.image.ResizeMethod.BILINEAR

                    A = A[0]  # remove batch dim; now x by y by z by channels
                    s = tf.shape(A)
                    # upsample y and z:
                    A = tf.image.resize_images(A, (s[1] * factor, s[2] * factor), method=method)
                    # upsample x:
                    A = tf.transpose(A, (3, 0, 1, 2))  # move channels dim to 0
                    A = tf.image.resize_images(A, (s[0] * factor, s[1] * factor), method=method)
                    # restore shape:
                    A = tf.transpose(A, (1, 2, 3, 0))[None]
                    return A

                net = upsample3D(net, 2)  # unlike paper, upsample before convs
                net = tf.layers.conv3d(net, filters=numfilters, kernel_size=kernel_size,
                                       strides=(1, 1, 1), padding='same')
                net = self.normalizing_layer(net)
                net = tf.nn.leaky_relu(net)

                net = tf.layers.conv3d(net, filters=numfilters, kernel_size=1,
                                       strides=(1, 1, 1), padding='same')  # kernel size 1
                net = self.normalizing_layer(net)
                return net

            def skip_block(net, numfilters=4, kernel_size=1):
                if numfilters == 0:  # no skip connections
                    return None
                else:
                    net = tf.layers.conv3d(net, filters=numfilters,
                                           kernel_size=kernel_size,
                                           strides=(1, 1, 1), padding='same')
                    net = self.normalizing_layer(net)
                    net = tf.nn.leaky_relu(net)
                    return net

            if len(self.numfilters_list) != len(self.numskipfilters_list):
                # the longer list will be truncated;
                print('Warning: length of numfilters_list and numskip_filters list not the same!')

            net = net_input
            print(net)

            skip_block_list = list()
            for numfilters, numskipfilters in zip(self.numfilters_list, self.numskipfilters_list):
                net = downsample_block(net, numfilters)
                print(net)
                skip_block_list.append(skip_block(net, numskipfilters))
            for numfilters, skip_block in zip(self.numfilters_list[::-1][:-1], skip_block_list[::-1][:-1]):
                if skip_block is not None:
                    # first pad skip block in case input doesn't have dims that are powers of 2:
                    skip_shape = tf.shape(skip_block)  # 1, x, y, z, numfilt
                    net_shape = tf.shape(net)
                    pad_x = net_shape[1] - skip_shape[1]
                    pad_y = net_shape[2] - skip_shape[2]
                    pad_z = net_shape[3] - skip_shape[3]
                    # handle odd numbers by using ceil:
                    skip_block = tf.pad(skip_block, [[0, 0],
                                                     [tf.to_int32(pad_x / 2), tf.to_int32(tf.ceil(pad_x / 2))],
                                                     [tf.to_int32(pad_y / 2), tf.to_int32(tf.ceil(pad_y / 2))],
                                                     [tf.to_int32(pad_z / 2), tf.to_int32(tf.ceil(pad_z / 2))],
                                                     [0, 0]])
                    net = tf.concat([net, skip_block], axis=4)
                net = upsample_block(net, numfilters)
                net = tf.nn.leaky_relu(net)
                print(net)
            # process the last layer separately, because no activation:
            net = upsample_block(net, self.numfilters_list[0])
            if not self.linear_DIP_output:
                net = tf.nn.leaky_relu(net)
            print(net)
            net = tf.squeeze(net)  # remove batch dimension, which is 1
            return net

        input_featmaps = 32
        # network input:
        # smallest power of 2 greater than the current dims (to allow skip connections):
        if self.DIP_make_pow2:
            side_kxy = np.int32(2 ** np.ceil(np.log(self.side_kxy) / np.log(2)))
            side_kz = np.int32(2 ** np.ceil(np.log(self.side_kz) / np.log(2)))
        else:
            side_kxy = self.side_kxy
            side_kz = self.side_kz

        if self.use_spatial_patching:
            # if you're using spatial patching, just choose a recon size that's a power of 2
            assert self.DIP_make_pow2 is False
            side_kxy, _, side_kz = self.recon_shape_full
            self.noisy_input = tf.Variable(np.random.rand(1, side_kxy, side_kxy, side_kz,
                                                          input_featmaps) * .1, dtype=tf.float32, trainable=False)
            ULcorner = self.spatial_batch_inds[:, 0]
            begin = tf.concat([[0], ULcorner, [0, 0]], axis=0)
            # cropping only x and y
            crop = tf.slice(self.noisy_input, begin=begin,
                            size=(1, self.xy_cap_n, self.xy_cap_n, side_kz, input_featmaps))
            net_input = crop
        else:
            self.noisy_input = tf.Variable(np.random.rand(1, side_kxy, side_kxy, side_kz,
                                                          input_featmaps) * .1, dtype=tf.float32, trainable=False)
            net_input = self.noisy_input

        with tf.variable_scope('DIP'):
            net = build_model(net_input=net_input)
        # adjust shape to match recon size:
        if self.DIP_make_pow2:
            # if made pow2, then the output shape is the same
            side_kxy_out = side_kxy
            side_kz_out = side_kz
        else:
            s = tf.shape(net)
            side_kxy_out = s[0]
            side_kz_out = s[2]
        start_xy = (side_kxy_out - self.side_kxy) // 2
        start_z = (side_kz_out - self.side_kz) // 2
        net = net[start_xy:start_xy + self.side_kxy, start_xy:start_xy + self.side_kxy, start_z:start_z + self.side_kz]

        # scale factor:
        net *= self.DIP_output_scale

        # use half of the channel dimension for real, half for imag
        real, imag = tf.split(net, num_or_size_splits=2, axis=-1)
        # sum across channel dimension
        self.DT_recon_r = tf.reduce_sum(real, -1)
        self.DT_recon_i = tf.reduce_sum(imag, -1)
        self.net = net

    def stochastic_stitch(self, sess, num_crop=1000, depad=35):
        # stochastic version of stitching for DIP and MS, whereby random crops are averaged together in their proper
        # location in the reconstruction; the larger num_crop, the more accurate;
        # num_crop=1000 takes roughly 155 sec on a tesla T4 GPU;
        crop_size_xy = self.xy_cap_n - depad * 2

        recon_full = np.zeros((self.xy_full_n, self.xy_full_n, self.side_kz), dtype=np.complex64)
        coverage = np.copy(recon_full)  # how many times each pixel was visited, to normalize by
        for i in range(num_crop):
            # get next crop:
            RI_r, RI_i, UL = sess.run([self.DT_recon_r, self.DT_recon_i, self.spatial_batch_inds])
            UL = UL[:, 0]
            RI = RI_r + 1j * RI_i
            RI += self.n_back
            RI_depad = RI[depad:-depad, depad:-depad, :]

            # populate recon_full with crop, and update coverage:
            recon_full[UL[0] + depad:UL[0] + depad + crop_size_xy,
            UL[1] + depad:UL[1] + depad + crop_size_xy] += RI_depad
            coverage[UL[0] + depad:UL[0] + depad + crop_size_xy,
            UL[1] + depad:UL[1] + depad + crop_size_xy] += 1

        return recon_full[depad:-depad, depad:-depad] / coverage[depad:-depad, depad:-depad]

    def V_to_RI(self, V):
        # since this needs to be computed a lot:
        if tf.contrib.framework.is_tensor(V):
            mod = tf
        else:
            mod = np
        return mod.sqrt(self.FDT_calibration_factor * V / (2 * np.pi * self.k_vacuum) ** 2 + self.n_back ** 2)

    def RI_to_V(self, RI):
        return (self.k_vacuum * 2 * np.pi) ** 2 * (RI ** 2 - self.n_back ** 2) / self.FDT_calibration_factor

    def tf_fftshift3(self, A):
        # 3D fftshift; apply fftshift to the last 3 dims
        s = tf.shape(A)
        s1 = s[-3] + 1
        s2 = s[-2] + 1
        s3 = s[-1] + 1
        A = tf.concat([A[..., s1 // 2:, :, :], A[..., :s1 // 2, :, :]], axis=-3)
        A = tf.concat([A[..., :, s2 // 2:, :], A[..., :, :s2 // 2, :]], axis=-2)
        A = tf.concat([A[..., :, :, s3 // 2:], A[..., :, :, :s3 // 2]], axis=-1)
        return A

    def tf_ifftshift3(self, A):
        # 3D ifftshift; apply ifftshift to the last 3 dims
        s = tf.shape(A)
        s1 = s[-3]
        s2 = s[-2]
        s3 = s[-1]
        A = tf.concat([A[..., s1 // 2:, :, :], A[..., :s1 // 2, :, :]], axis=-3)
        A = tf.concat([A[..., :, s2 // 2:, :], A[..., :, :s2 // 2, :]], axis=-2)
        A = tf.concat([A[..., :, :, s3 // 2:], A[..., :, :, :s3 // 2]], axis=-1)
        return A

    def tf_fftshift2(self, A):
        # 2D fftshift; apply fftshift to the last two dims
        s = tf.shape(A)
        s1 = s[-2] + 1
        s2 = s[-1] + 1
        A = tf.concat([A[..., s1 // 2:, :], A[..., :s1 // 2, :]], axis=-2)
        A = tf.concat([A[..., :, s2 // 2:], A[..., :, :s2 // 2]], axis=-1)
        return A

    def tf_ifftshift2(self, A):
        # 2D ifftshift; apply ifftshift to the last two dims
        s = tf.shape(A)
        s1 = s[-2]
        s2 = s[-1]
        A = tf.concat([A[..., s1 // 2:, :], A[..., :s1 // 2, :]], axis=-2)
        A = tf.concat([A[..., :, s2 // 2:], A[..., :, :s2 // 2]], axis=-1)
        return A

    def tf_gather_nd3(self, params, indices):
        # gather_nd assuming 3 dims in params; for gathering xyz_caps;
        # (I had issues with tf.gather_nd when using a GPU);
        s = tf.shape(params)
        params = tf.reshape(params, [-1])
        indices = indices[..., 2] + indices[..., 1] * s[2] + indices[..., 0] * s[1] * s[2]
        return tf.gather(params, indices)