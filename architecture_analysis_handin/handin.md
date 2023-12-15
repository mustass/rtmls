## Architectural analysis of the wgpu + egui + winit module

### Used crates/modules 

From the name it is quite obvious that we will use **wgpu**, **egui** and **winit**. However, we will also use **crossbeam**. 

**wgpu** is used to render both the gui (i.e. the Control Panel window) and the triangle render window. Furthermore, we also use wgpu to render the triangle inside the window itself. 

**egui** is used to create the control panel and its widgets. The library also handles listening to the events coming from the widgets. 

**winit** is used to create the windows themselves. The library also handles the events concerning to what happens to the window: resizing, dragging, clicking, scrolling, keyboard input, etc. Pretty much anything the window can be expected to react to.  

**crossbeam** is used to create a channel of communicaton between the Control Panel and the Render Engine

### The setup

We will look at it from a modules point of view. As such, we have 3 main ones:

* Main `run()`  function that subsequently calls and blocks on an asynch  `run_loop()` function.
* `render_engine`
* `control_panel` 

In the following we try and visualize what's going on. 
**Little drawing**
![alt text](architecture_drawing.png "Architecture Overview")

And with this in mind, we can go through the data flow in the code. 
Basically, after the initializations we need to communicate two things: (1) what events we recieve in the two windows from **winit** and (2) what events we recieve from the Control Panel (**egui**). The fun thing is that (2) relies on (1) as our gui lives inside the window. 

The main thread is where a lot of the architecture action is located. Here we initialize the objects, launch the additional thread for the Render Engine and run the EventLoop handling our windows. It would be tedious to go through all the movement of the objects in words, so the keen reader is directed towards the drawing instead. 

The thread running the Render Engine has mostly its own data. The interesting part to consider here is that we need to let it use the channel of communication between the main thread and its own thread. This is done by instantiating that said channel which returns a receiver, transmitter pair. The main thread moves ownership of the receiver to the thread running the render engine as well as the render engine object itself. This thread will now be able to receive communication from the main thread about what is happening to the window it is rendering the triangle into as well as the information passed by the user of the gui through its own window. 




* * *

