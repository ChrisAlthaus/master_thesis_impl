import ipywidgets as widgets
import asyncio
from IPython.display import Javascript, display
from ipywidgets import widgets, Button, HBox, Label
import functools

def displaySliderExecNexts(tresh):
    class Timer:
        def __init__(self, timeout, callback):
            self._timeout = timeout
            self._callback = callback
            self._task = asyncio.ensure_future(self._job())

        async def _job(self):
            await asyncio.sleep(self._timeout)
            self._callback()

        def cancel(self):
            self._task.cancel()

    def debounce(wait):
        """ Decorator that will postpone a function's
            execution until after `wait` seconds
            have elapsed since the last time it was invoked. """
        def decorator(fn):
            timer = None
            def debounced(*args, **kwargs):
                nonlocal timer
                def call_it():
                    print("call",args)
                    fn(*args, **kwargs)
                if timer is not None:
                    timer.cancel()
                timer = Timer(wait, call_it)
            return debounced
        return decorator

    slider = widgets.FloatSlider(min=0, max =1, step= 0.01, value = 1)


    @debounce(0.2)
    def on_button_clicked(val):
        treshsave = val
        display(Javascript('IPython.notebook.execute_cells_below()'))

    slider.observe(on_button_clicked, names="value")
    display(slider)


def sliderButtonExecNexts():
    slider = widgets.FloatSlider(min=0, max =1, step= 0.01, value = 1)
    button = widgets.Button(description="Click Me!")

    def on_button_clicked(b):
        print(slider.value)
        #display(Javascript('IPython.notebook.execute_cells_below()'))

    button.on_click(functools.partial(on_button_clicked))
    display(HBox([slider, button]))
    return slider

def floatslider():
    slider = widgets.FloatSlider(min=0, max =1, step= 0.01, value = 1)
    
    display(HBox([Label('Select the treshold: '), slider]))
    return slider