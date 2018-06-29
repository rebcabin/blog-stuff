from contextlib import contextmanager

@contextmanager
def keep_plots_open(keep_show_open_on_exit=True, even_when_error=True):
    '''
    To continue executing code when plt.show() is called
    and keep the plot on displaying before this context manager exits
    (even if an error caused the exit).
    '''
    import matplotlib.pyplot
    show_original = matplotlib.pyplot.show

    def show_replacement(*args, **kwargs):
        kwargs['block'] = False
        show_original(*args, **kwargs)

    matplotlib.pyplot.show = show_replacement

    pylab_exists = True

    try:
        import pylab
    except ImportError:
        pylab_exists = False
    if pylab_exists:
        pylab.show = show_replacement

    try:
        yield
    except Exception as err:
        if keep_show_open_on_exit and even_when_error:
            print("*********************************************")
            print("Error early edition while waiting for show():")
            print("*********************************************")
            import traceback
            print(traceback.format_exc())
            show_original()
            print("*********************************************")
            raise
    finally:
        matplotlib.pyplot.show = show_original
        if pylab_exists:
            pylab.show = show_original

    if keep_show_open_on_exit:
        show_original()

# ***********************
# Running example
# ***********************
import pylab as pl
import time
if __name__ == '__main__':
    with keep_plots_open():
        pl.figure('a')
        pl.plot([1,2,3], [4,5,6])
        pl.plot([3,2,1], [4,5,6])
        pl.show()

        pl.figure('b')
        pl.plot([1,2,3], [4,5,6])
        pl.show()

        time.sleep(1)
        print('...')
        time.sleep(1)
        print('...')
        time.sleep(1)
        print('...')

        this_will_surely_cause_an_error
