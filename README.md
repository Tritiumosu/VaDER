VaDER (Voice and Data Enhanced Radio) is a learning project, to help me learn Python and better understand the function (and limitations) of CAT control of a physical radio over USB/Serial. 
Currently, this project is focused on the radio I own - a Yaesu FT-991A - but I hope to be able to include CAT libraries for other models of radio once the basic functionality is established. 

Long-term goals include an interface that doesn't look like crap, and useful controls to allow for new users to rapidly get comfortable and make use of their equipment. 

I am not a coding professional, so there will be heavy use of AI assistants and probably more than one dumb question asked to friends and Elmers with more programming and radio experience. 
73 - W8TSB

Features included so far: 
- CAT Control file established for FT-991A with selectable port/rate/stop setting
- Ugly displays for frequency, audio RMS, S-Meter, RF Power, and Mode
- Rudimentary band scanning for 2m, 10m, and 20m ham bands
- Manual log that doesn't save the data anywhere yet
- Honest-to-goodness Python version of the WSJT-X FT8 decoder (with configurable input)

Features that are not included or do not really work great
- The aforementioned logging problem
- PTT button does trigger the PTT on the radio, but isn't super useful unless you just wanna talk.
- Any other part of the FT8 exchange process, including transmitting any kind of FT8 tones
- Most other useful radio functions
