# Differentiable Electrochemistry simulation of voltammetry in weakly supported media 

This simulator is arguably the most complicated simulator among all five simulators. Consider the following electrochemical reaction:

A + e<sup>-</sup> = B 

There is however, consideration of suppporting electrolyte M and N, with charge z<sub>M</sub> and z<sub>N</sub>. In a weakly supported media where the concentration of supporting electrolyte is low (the support ratio is low), the migration transport effect and electric field must be considered. In this case, we have solved Nernst-Planck-Poission equation, with Butler-Volmer or Marcus-Hush-Chidsey kientics with linear diffusion.