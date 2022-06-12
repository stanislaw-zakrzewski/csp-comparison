def visualize_csp_filters(bands, band_data, n_csp_components):
    for index, band in enumerate(bands):
        band_name = '{}Hz - {}Hz'.format(band[0], band[1])
        print(band_name)
        band_data[index][2].plot_patterns(band_data[index][3], components=range(n_csp_components), ch_type='eeg',
                                          units='Patterns (AU)',
                                          size=1.5)
