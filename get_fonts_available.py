# based on https://stackoverflow.com/questions/45877746/changing-fonts-in-matplotlib

import  matplotlib.font_manager

def get_fname(fname):
    try:
        name = matplotlib.font_manager.FontProperties(fname=fname).get_name()
        return name
    except RuntimeError:
        return ''
    

flist = matplotlib.font_manager.get_fontconfig_fonts()
names = sorted({get_fname(fname) for fname in flist})
print(names)

# Try using some fonts in a graph.
# First, clear cache so recently added fonts are present.
matplotlib.font_manager._rebuild()

for font in [
    'SchulbuchNord Normal',
    'Frutiger CE 55 Roman', 
    'Futura LT', 'Futura Next', 'Avenir Next LT Pro', 
    'Roboto', 'Univers LT Std', 
    'Helvetica', 'Helvetica Neue', 'Helvetica Now Text ', 'Helvetica Neue LT Std',
    'Karla', 'Lato', 
    'Montserrat', 'NexaRegular',  
    ]:
    
    print('\n\n ', font)
    del(plt)
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font
    # big text
    family_name = font
    font = {'family' : 'sans-serif',
            'sans-serif': font,
            'weight' : 'regular',
            'size'   : 16}
    plt.rc('font', **font)
   
    plt.figure(figsize=(11,6))
    plt.plot(np.arange(0, 100, 5), np.arange(0, 200, 10))
    plt.plot(np.arange(0, 500, 5), np.arange(0, 1000, 10))

    ax = plt.gca()
    ax.set_title(f'The quick brown fox jumps over the lazy dog {family_name}')#, fontname=font)
   
    plt.legend(['Line # 1', 'Line # 2'])
    plt.ylim((0, 2650))
    plt.show()

