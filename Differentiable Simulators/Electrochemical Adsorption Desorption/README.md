# Differentiable Electrochemical Simulation for Voltammetry with Adsorption/Desorption 


The electrochemical reaction is:

A + e<sup>-</sup> = B

A<sub>ads</sub> + e = B<sub>ads</sub>

where both the solution and adsorbed species are electroactive. They have, however, different reaction rate constants and equilibrium. In addition, the interconversion between the adsorbed/solution phase species must be modeled. 

Thus, there are at least *10* parameters that are differentiable in this case. A graphical view of this problem is shown below.  

![AdsorptionMechanism](AdsorptionMechanism.png)


See below an example voltammogram where only adsorptive is electrochemically active, and its comparison with analytical equation.  
![FullyAdsorptive](<Fully adsorptive.png>)