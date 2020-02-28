(* ::Package:: *)

(* ::Section:: *)
(*Ensembling Random Features - Functions for Plots*)


(* ::Text:: *)
(*In this file, we give the main functions for plotting the generalisation error of a random features system as well as the terms appearing in the bias variance decomposition. There are also the ones needed for the various comparisons between ensembling and regularisation.*)
(*Throughout we define: \[Psi]1=P/D=(Number of features)/(Data dimension) and \[Psi]2=N/D=(Training set size)/(data dimension)*)


(*Please Add here the location of the file EnsembleRFErrorCode.m assumed that is in the same repository as tis file*)
Get[StringJoin[NotebookDirectory[],"EnsembleRFErrorCode.m"]]


(* ::Subsubsection:: *)
(*Function to Generate Plot*)


ClearAll[genPlot];
Options[genPlot]:={"pts"->Null,"plotRange"->{{-1,2},{0,1.5}},"xaxisLabel"->None,"yaxisLabel"->None,"leg"->Null,"colors"->0,"legPos"->{.82,.8},"dashed"->0 };
genPlot[OptionsPattern[]]:= Module[{plot,pts=OptionValue["pts"],xaxisLabel=OptionValue["xaxisLabel"],
yaxisLabel=OptionValue["yaxisLabel"],colors=OptionValue["colors"],dashed=OptionValue["dashed"],leg=OptionValue["leg"],plotRange=OptionValue["plotRange"],
legPos=OptionValue["legPos"],style,plotStyle},

If[colors==0,
Print[colors];
colors=ColorData[97,"ColorList"];
colors=colors[[1;;Length[pts]]];
];

If[dashed==0,
plotStyle=colors;
];

If[Length[dashed]>0,
plotStyle=Partition[Riffle[colors,dashed],2]
];
style={FontSize->16,FontWeight->Bold,FontColor->Black};
plot=Labeled[Show[ListLinePlot[pts,PlotRange->plotRange,PlotStyle->plotStyle,AxesLabel->{None,yaxisLabel},LabelStyle->style],Graphics[Inset[LineLegend[colors,leg,LabelStyle->style],Scaled@legPos]],ImageSize->Large],xaxisLabel,{{Bottom,Right}},LabelStyle->style];
plot
]


(* ::Subsubsection::Closed:: *)
(*Functions to Compare Performance MoreVSBig *)


ClearAll[moreVsbig];
moreVsbig::usage = "more vs big is used to compare the performances of two systems: one assembled over k subsystems and one with 1 system having k times more hidden units.
  moreVsbig[k,\[Psi]1,\[Psi]2,\[Lambda],F1_:1.0,FStar_:0.1,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]
  Returns the value of the error in both cases";
moreVsbig[\[Psi]1_,\[Psi]2_, \[Lambda]_,k_,F1_: 1.0, FStar_: 0.0, \[Tau]_: 0.0, \[Mu]1_: 0.5, \[Mu]Star_: 0.3014051374945435`] := Module[{int,errBig, errMore,errInf,errBaseline},
   errBig = ErrorVanilla[k*\[Psi]1,\[Psi]2,\[Lambda],F1,FStar,\[Tau],\[Mu]1,\[Mu]Star][[2]];
   int=ErrorMixed[\[Psi]1, \[Psi]2, \[Lambda],k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];
   errBaseline=int[[6]];
   {errMore,errInf}=int[[6,7]];
   {\[Lambda],\[Psi]1,\[Psi]2,k,errBig, errMore,errInf,errBaseline}
   ];

ClearAll[plotmoreVsbig];
Options[plotmoreVsbig]:={"\[Psi]1"->1.5,"k"->2,"\[Lambda]"->10^-5,"plot"->True, "NbPoints"->30, "minVal"-> -1, "maxVal"->2.0,
"F1"-> 1.0, "FStar"-> 0.0, "\[Tau]"-> 0.0, "\[Mu]1"-> 0.5, "\[Mu]Star"-> 0.3014051374945435`}
plotmoreVsbig::usage = "plotmoreVsbig is a function that returns two lists of points two compare the performances of a bigger networks with respect to a smaller network. 
If true, also plots the points
options are: \[Psi]2\[Rule]1.5,k\[Rule]2,\[Lambda]\[Rule]\!\(\*SuperscriptBox[\(10\), \(-5\)]\) Plot:True, NbPoints: 30,minVal: -1 , maxVal:2.0,
F1:1.0, FStar:0.0, \[Tau]:0.0, \[Mu]1: Relu, \[Mu]Star: ReLU";
plotmoreVsbig[OptionsPattern[]]:=Module[{
plot,colors,xVals,
\[Psi]1=OptionValue["\[Psi]1"],k=OptionValue["k"],\[Lambda]=OptionValue["\[Lambda]"],
plotBol=OptionValue["plot"],
F1=OptionValue["F1"],FStar=OptionValue["FStar"],\[Tau]=OptionValue["\[Tau]"],\[Mu]1=OptionValue["\[Mu]1"],\[Mu]Star=OptionValue["\[Mu]Star"],
minVal=OptionValue["minVal"],maxVal=OptionValue["maxVal"],NbPoints=OptionValue["NbPoints"],
ptsM,ptsV,ptsB,err,leg,axesLabel
},
xVals=N@Subdivide[minVal,maxVal,NbPoints-1];
ptsM={};ptsV={};ptsB={};
Do[
{
err=ErrorMixed[\[Psi]1,10^x,\[Lambda],k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];
AppendTo[ptsB,{Log10[err[[3]]],err[[-3]]}];
AppendTo[ptsM,{Log10[err[[3]]],err[[-2]]}];
err=ErrorVanilla[k*\[Psi]1,10^x,\[Lambda],F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];
AppendTo[ptsV,{Log10[err[[3]]],err[[-1]]}];
},
{x,xVals}
];
ptsM=Select[ptsM,#[[2]]>0&];
ptsV=Select[ptsV,#[[2]]>0&];
ptsB=Select[ptsB,#[[2]]>0&];
colors=ColorData[97,"ColorList"];
colors=colors[[1;;3]];
leg={"Baseline k=1","Overparametrize",StringJoin["Ensemble k=",ToString[k]]};
axesLabel={"Log10[P/D]","Generalisation\n      Error"};
If[plotBol,

plot=genPlot["pts"->{ptsV,ptsM,ptsB},"xaxisLabel"->"\!\(\*SubscriptBox[\(Log\), \(10\)]\)(\!\(\*FractionBox[\(P\), \(N\)]\))","yaxisLabel"->"Generalisation \n   Error","leg"-> leg];
Return[{ptsV,ptsM,ptsB,plot}];
];
{ptsV,ptsM,ptsB}
];


(* ::Subsubsection::Closed:: *)
(*Function to Plot Generalisation Error Vanilla Case*)


ClearAll[plotErrorVanilla];
Options[plotErrorVanilla]:={"\[Psi]2"->1.5,"plot"->True, "NbPoints"->30, "minVal"-> -1, "maxVal"->2.0,"\[Lambda]"->0.0,"\[Psi]1"->0.0,
"F1"-> 1.0, "FStar"-> 0.0, "\[Tau]"-> 0.0, "\[Mu]1"-> 0.5, "\[Mu]Star"-> 0.3014051374945435`}
plotError::usage = "plotError returns the points for the mixed error, the vanilla error and k-> \[Infinity].
The evolution is as a function of P/N=\!\(\*FormBox[\(\*FractionBox[\(Number\\\ of\\\ features\), \(Training\\\ set\\\ size\)]\\\ if\\\ no\\\ value\\\ is\\\ passed\\\ and\\\ \[Lambda]\\\ value\\\ is\\\ given . \\\\nTo\\\ plot\\\ as\\\ a\\\ function\\\ of\\\ \[Lambda], \\\ pass\\\ a\\\ value\\\ of\\\ P/N\\\ \(\(used\)\(.\)\(\\\ \\\ \)\)\),
TraditionalForm]\)
options are: \[Psi]2,k,lambda:0 (means that lambda evolves),\[Psi]1->0.0, Plot:True, NbPoints: 30,minVal: -1 , maxVal:2.0,
F1:1.0, FStar:0.0, \[Tau]:0.0, \[Mu]1: Relu, \[Mu]Star: ReLU";

plotErrorVanilla[OptionsPattern[]]:=Module[{
\[Psi]2=OptionValue["\[Psi]2"],
\[Lambda]=OptionValue["\[Lambda]"],\[Psi]1=OptionValue["\[Psi]1"],
ptsMore,ptsBig,plot,colors,xVals,plotBol=OptionValue["plot"],
F1=OptionValue["F1"],FStar=OptionValue["FStar"],\[Tau]=OptionValue["\[Tau]"],\[Mu]1=OptionValue["\[Mu]1"],\[Mu]Star=OptionValue["\[Mu]Star"],
minVal=OptionValue["minVal"],maxVal=OptionValue["maxVal"],NbPoints=OptionValue["NbPoints"],
ptsV,err,axesLabel,leg,ii
},

xVals=N@Subdivide[minVal,maxVal,NbPoints-1];


If[\[Psi]1==0.0&&\[Lambda]==0,Print["Please give a value of \[Lambda] or a value of \[Psi]1"];Return[$Failed];];
ii=1;
ptsV={};
If[\[Psi]1==0.0,

Do[
{

err=ErrorVanilla[10^x*\[Psi]2,\[Psi]2,\[Lambda],F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];
AppendTo[ptsV,{Log10[err[[2]]/err[[3]]],err[[-1]]}];
ii=ii+1;
},
{x,xVals}
];
axesLabel={"Log10[P/N]","Generalisation\n      Error"};,
Do[
{

err=ErrorVanilla[\[Psi]1,\[Psi]2,10^x,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];

AppendTo[ptsV,{Log10[err[[1]]],err[[-1]]}];
ii=ii+1;
},
{x,xVals}
];
axesLabel={"Log10[\[Lambda]]","Generalisation\n      Error"};
];

ptsV=Select[ptsV,#[[2]]>0&];
Print[Length[ptsV]];
colors=ColorData[97,"ColorList"];
colors=colors[[1]];
leg={"Vanilla k=1"};
If[plotBol,
plot=genPlot["pts"->{ptsV},"xaxisLabel"->"\!\(\*SubscriptBox[\(Log\), \(10\)]\)(\!\(\*FractionBox[\(P\), \(N\)]\))","yaxisLabel"->"Generalisation \n   Error","leg"-> leg];
Return[{ptsV,plot}];,
Return[ptsV]
];
ptsV
];


(* ::Subsubsection::Closed:: *)
(*Function to Plot Generalisation Error*)


ClearAll[plotError];
Options[plotError]:={"\[Psi]2"->1.5,"k"->2,"plot"->True, "NbPoints"->30, "minVal"-> -1, "maxVal"->2.0,"\[Lambda]"->0.0,"\[Psi]1"->0.0,"pts"->{0},
"F1"-> 1.0, "FStar"-> 0.0, "\[Tau]"-> 0.0, "\[Mu]1"-> 0.5, "\[Mu]Star"-> 0.3014051374945435`,"legPos"->{0.825,0.8},"pts"->{0},"plotRange"->{0,1.5}}
plotError::usage = "plotError returns the points for the mixed error, the vanilla error and k-> \[Infinity].
The evolution is as a function of P/N=\!\(\*FormBox[\(\*FractionBox[\(Number\\\ of\\\ features\), \(Training\\\ set\\\ size\)]\\\ if\\\ no\\\ value\\\ is\\\ passed\\\ and\\\ \[Lambda]\\\ value\\\ is\\\ given . \\\\nTo\\\ plot\\\ as\\\ a\\\ function\\\ of\\\ \[Lambda], \\\ pass\\\ a\\\ value\\\ of\\\ P/N\\\ \(\(used\)\(.\)\(\\\ \\\ \)\)\),
TraditionalForm]\)
options are: \[Psi]2,k,lambda:0 (means that lambda evolves),\[Psi]1->0.0, Plot:True, NbPoints: 30,minVal: -1 , maxVal:2.0,
F1:1.0, FStar:0.0, \[Tau]:0.0, \[Mu]1: Relu, \[Mu]Star: ReLU";

plotError[OptionsPattern[]]:=Module[{plotRange=OptionValue["plotRange"],
\[Psi]2=OptionValue["\[Psi]2"],k=OptionValue["k"],
\[Lambda]=OptionValue["\[Lambda]"],\[Psi]1=OptionValue["\[Psi]1"],
ptsMore,ptsBig,plot,colors,xVals,plotBol=OptionValue["plot"],
F1=OptionValue["F1"],FStar=OptionValue["FStar"],\[Tau]=OptionValue["\[Tau]"],\[Mu]1=OptionValue["\[Mu]1"],\[Mu]Star=OptionValue["\[Mu]Star"],
minVal=OptionValue["minVal"],maxVal=OptionValue["maxVal"],NbPoints=OptionValue["NbPoints"],legPos=OptionValue["legPos"],pts=OptionValue["pts"],
ptsM,ptsV,ptsInf,err,xaxesLabel,yaxesLabel,leg,ii
},
If[Length[pts]==1,
xVals=N@Subdivide[minVal,maxVal,NbPoints-1];,
xVals=#[[1]]&/@pts[[1]];];

ptsM={};ptsV={};ptsInf={};
If[\[Psi]1==0.0&&\[Lambda]==0,Print["Please give a value of \[Lambda] or a value of \[Psi]1"];Return[$Failed];];
ii=1;
If[\[Psi]1==0.0,
{
Do[
{
If[Length[pts]==1,
err=ErrorMixed[10^x*\[Psi]2,\[Psi]2,\[Lambda],k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];,
err=ErrorMixed[10^x*\[Psi]2,\[Psi]2,\[Lambda],k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star,"pts"-> pts[[All,ii]]];];
AppendTo[ptsV,{Log10[err[[2]]/err[[3]]],err[[-3]]}];
AppendTo[ptsM,{Log10[err[[2]]/err[[3]]],err[[-2]]}];
AppendTo[ptsInf,{Log10[err[[2]]/err[[3]]],err[[-1]]}];
ii=ii+1;
},
{x,xVals}
];
xaxesLabel="Log10[P/N]";
yaxesLabel="Generalisation\n      Error";
},
{
Do[
{
If[Length[pts]==1,
err=ErrorMixed[\[Psi]1,\[Psi]2,10^x,k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star];
AppendTo[ptsV,{Log10[err[[1]]],err[[-3]]}];
AppendTo[ptsM,{Log10[err[[1]]],err[[-2]]}];
AppendTo[ptsInf,{Log10[err[[1]]],err[[-1]]}];,
err=ErrorMixed[\[Psi]1,\[Psi]2,10^x,k,F1,FStar,\[Tau],\[Mu]1,\[Mu]Star,"pts"-> pts[[All,ii]]];
AppendTo[ptsV,{Log10[err[[1]]],err[[-3]]}];
AppendTo[ptsM,{Log10[err[[1]]],err[[-2]]}];
AppendTo[ptsInf,{Log10[err[[1]]],err[[-1]]}];
];

ii=ii+1;},
{x,xVals}
];
xaxesLabel="Log10[\[Lambda]]";
yaxesLabel="Generalisation\n      Error";
}
];
ptsM=Select[ptsM,#[[2]]>0&];
ptsV=Select[ptsV,#[[2]]>0&];
ptsInf=Select[ptsInf,#[[2]]>0&];
colors=ColorData[97,"ColorList"];
colors=colors[[1;;3]];
leg={"Vanilla k=1",StringJoin["Ensembled k=",ToString[k]],"Ensembled k=\[Infinity]"};
If[plotBol,
plot=genPlot["pts"->{ptsV,ptsM,ptsInf},"xaxisLabel"->xaxesLabel,"yaxisLabel"->"Generalisation \n   Error","leg"-> leg,"colors"->colors,"legPos"->legPos,"plotRange"-> plotRange];

Return[{ptsV,ptsM,ptsInf,plot}];
];
{ptsV,ptsM,ptsInf}
];


(* ::Subsubsection::Closed:: *)
(*Functions to Plot Term by Term*)


ClearAll[plotTermbyTermEvol];
Options[plotTermbyTermEvol]:={"\[Psi]2"->1.5,"plot"->True, "NbPoints"->30, "minVal"-> -1, "maxVal"->2.0,"\[Lambda]"->0.0,"\[Psi]1"->0.0,
"\[Mu]1"-> 0.5, "\[Mu]Star"-> 0.3014051374945435`}
plotTermbyTermEvol::usage = "plotTermbyTermEvol returns the points for the different terms appering in the generalisation error.
The evolution is as a function of P/N=\!\(\*FormBox[\(\*FractionBox[\(Number\\\ of\\\ features\), \(Training\\\ set\\\ size\)]\\\ if\\\ no\\\ value\\\ is\\\ passed\\\ and\\\ \[Lambda]\\\ value\\\ is\\\ given . \\\\nTo\\\ plot\\\ as\\\ a\\\ function\\\ of\\\ \[Lambda], \\\ pass\\\ a\\\ value\\\ of\\\ P/N\\\ \(\(used\)\(.\)\(\\\ \\\ \)\)\),
TraditionalForm]\)
options are: \[Psi]2,k,lambda:0 (means that lambda evolves),\[Psi]1->0.0, Plot:True, NbPoints: 30,minVal: -1 , maxVal:2.0,
F1:1.0, FStar:0.0, \[Tau]:0.0, \[Mu]1: Relu, \[Mu]Star: ReLU";


plotTermbyTermEvol[OptionsPattern[]] := Module[{
\[Psi]2=OptionValue["\[Psi]2"],
\[Lambda]=OptionValue["\[Lambda]"],\[Psi]1=OptionValue["\[Psi]1"],
ptsMore,ptsBig,plot,colors,xVals,plotBol=OptionValue["plot"],\[Mu]1=OptionValue["\[Mu]1"],\[Mu]Star=OptionValue["\[Mu]Star"],
minVal=OptionValue["minVal"],maxVal=OptionValue["maxVal"],NbPoints=OptionValue["NbPoints"]
,axesLabel,terms,
\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6,leg},

{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3}={{},{},{}};
{\[CapitalPsi]4,\[CapitalPsi]5}={{},{}};
\[CapitalPsi]6={};
xVals=N@Subdivide[minVal,maxVal,NbPoints-1];

If[\[Psi]1==0.0&&\[Lambda]==0,Print["Please give a value of \[Lambda] or a value of \[Psi]1"];Return[$Failed];];

If[\[Psi]1==0.0,
{
Do[
{
terms=getVanillaTerms[\[Psi]2*10^x,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]1,{x,terms[[1]]}];
AppendTo[\[CapitalPsi]2,{x,terms[[2]]}];
AppendTo[\[CapitalPsi]3,{x,terms[[3]]}];
terms=getMixedTerms[\[Psi]2*10^x,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]4,{x,terms[[1]]}];
AppendTo[\[CapitalPsi]5,{x,terms[[2]]}];
terms=getPsi4[\[Psi]2*10^x,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]6,{x,terms}];
},
{x,xVals}
];
axesLabel={"Log10[P/N]"};
},
{
Do[
{
terms=getVanillaTerms[\[Psi]1,\[Psi]2,10^x,\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]1,{x,terms[[1]]}];
AppendTo[\[CapitalPsi]2,{x,terms[[2]]}];
AppendTo[\[CapitalPsi]3,{x,terms[[3]]}];
terms=getMixedTerms[\[Psi]1,\[Psi]2,10^x,\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]4,{x,terms[[1]]}];
AppendTo[\[CapitalPsi]5,{x,terms[[2]]}];
terms=getPsi4[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
AppendTo[\[CapitalPsi]6,{x,terms}];
},
{x,xVals}
];
axesLabel={"Log10[\[Lambda]]"};
}
];
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6}=Select[#,#[[2]]>0&]&/@{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6};
colors=ColorData[97,"ColorList"];
colors=colors[[1;;6]];
leg={"\[CapitalPsi]1","\[CapitalPsi]2","\[CapitalPsi]3","\[CapitalPsi]4","\[CapitalPsi]5","\[CapitalPsi]6"};
If[plotBol,
plot=genPlot["pts"->pts,"xaxisLabel"->"\!\(\*SubscriptBox[\(Log\), \(10\)]\)(\!\(\*FractionBox[\(P\), \(N\)]\))","yaxisLabel"->None,"leg"-> leg];
Return[{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6,plot}];
];
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6}
 ];


(* ::Subsubsection::Closed:: *)
(*Functions to Plot Bias and Variance*)


 ClearAll[plotBiasVarRatioEvol];
Options[plotBiasVarRatioEvol]:={"\[Psi]2"->1.5,"k"->2,"plot"->True, "NbPoints"->30, "minVal"-> -1, "maxVal"->2.0,"\[Lambda]"->0.0,"\[Psi]1"->0.0,"plotRange"->{Full,{0,1.5}},
"F1"-> 1.0, "FStar"-> 0.0, "\[Tau]"-> 0.0, "\[Mu]1"-> 0.5, "\[Mu]Star"-> 0.3014051374945435`,"pts"->{0},"legPos"->{.82,.8}}
plotBiasVarRatioEvol::usage = "plotTermbyTermEvol returns the points for the different terms appering in the generalisation error.
The evolution is as a function of P/N=\!\(\*FormBox[\(\*FractionBox[\(Number\\\ of\\\ features\), \(Training\\\ set\\\ size\)]\\\ if\\\ no\\\ value\\\ is\\\ passed\\\ and\\\ \[Lambda]\\\ value\\\ is\\\ given . \\\\nTo\\\ plot\\\ as\\\ a\\\ function\\\ of\\\ \[Lambda], \\\ pass\\\ a\\\ value\\\ of\\\ P/N\\\ \(\(used\)\(.\)\(\\\ \\\ \)\)\),
TraditionalForm]\)
options are: \[Psi]2,k,lambda:0 (means that lambda evolves),pts: if you have the various \[CapitalPsi] terms already computed or not,\[Psi]1->0.0, Plot:True, NbPoints: 30,minVal: -1 , maxVal:2.0,
F1:1.0, FStar:0.0, \[Tau]:0.0, \[Mu]1: Relu, \[Mu]Star: ReLU";  

plotBiasVarRatioEvol[OptionsPattern[]] := Module[{
\[Psi]2=OptionValue["\[Psi]2"],k=OptionValue["k"],
\[Lambda]=OptionValue["\[Lambda]"],\[Psi]1=OptionValue["\[Psi]1"],
pts=OptionValue["pts"],plot,colors,xVals,plotBol=OptionValue["plot"],
F1=OptionValue["F1"],FStar=OptionValue["FStar"],\[Tau]=OptionValue["\[Tau]"],\[Mu]1=OptionValue["\[Mu]1"],\[Mu]Star=OptionValue["\[Mu]Star"],
minVal=OptionValue["minVal"],maxVal=OptionValue["maxVal"],NbPoints=OptionValue["NbPoints"],plotRange=OptionValue["plotRange"],legPos=OptionValue["legPos"]
,xaxesLabel,terms,leg,
\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6,
resPsiV,resPsiM,varIn,varData,varNoise,bias,tot},
If[\[Psi]1==0.0&&\[Lambda]==0,Print["Please give a value of \[Lambda] or a value of \[Psi]1"];Return[$Failed];];

If[\[Psi]1==0.0,xaxesLabel="\!\(\*SubscriptBox[\(Log\), \(10\)]\)(P/N)"];
If[\[Lambda]==0.0,xaxesLabel="Log10[\[Lambda]]"];

If[Length[pts]==0,
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6}=plotTermbyTermEvol["\[Psi]2"->\[Psi]2,"plot"->False, "NbPoints"->NbPoints, "minVal"-> minVal, "maxVal"->maxVal,"\[Lambda]"->\[Lambda],"\[Psi]1"->\[Psi]1, "\[Mu]1"-> \[Mu]1, "\[Mu]Star"-> \[Mu]Star],
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[CapitalPsi]6}=pts
];
\[CapitalPsi]1=Select[\[CapitalPsi]1,#[[2]]>0.0&];
\[CapitalPsi]2=Select[\[CapitalPsi]2,#[[2]]>0.0&];
\[CapitalPsi]3=Select[\[CapitalPsi]3,#[[2]]>0.0&];
\[CapitalPsi]4=Select[\[CapitalPsi]4,#[[2]]>0.0&];
\[CapitalPsi]5=Select[\[CapitalPsi]5,#[[2]]>0.0&];
\[CapitalPsi]6=Select[\[CapitalPsi]6,#[[2]]>0.0&];
xVals=#[[1]]&/@\[CapitalPsi]1;
varIn=Table[{xVals[[i]],1/k F1^2 (\[CapitalPsi]2[[i]][[2]]-\[CapitalPsi]4[[i]][[2]])},{i,1,Length[\[CapitalPsi]4]}];
varNoise=Table[{xVals[[i]],(\[Tau]^2/k (\[CapitalPsi]3[[i]][[2]]-\[CapitalPsi]5[[i]][[2]])+\[Tau]^2 \[CapitalPsi]5[[i]][[2]])},{i,1,Length[\[CapitalPsi]5]}];
varData=Table[{xVals[[i]],F1^2 \[CapitalPsi]4[[i]][[2]]-\[CapitalPsi]6[[i]][[2]]},{i,1,Length[\[CapitalPsi]6]}];
bias=Table[{xVals[[i]],F1^2 (1-2 \[CapitalPsi]1[[i]][[2]])+\[CapitalPsi]6[[i]][[2]]},{i,1,Length[\[CapitalPsi]6]}];
tot=Table[{xVals[[i]],bias[[i]][[2]]+varIn[[i]][[2]]+varData[[i]][[2]]+varNoise[[i]][[2]]},{i,1,Length[\[CapitalPsi]6]}];
If[plotBol,
colors=ColorData[97,"ColorList"];
colors=AppendTo[colors[[1;;4]],Black];
leg={"Initialisation Variance","Noise Variance","Data Variance","Bias","Generalisation Error"};
plot=genPlot["pts"->{varIn,varNoise,varData,bias,tot},"xaxisLabel"->"\!\(\*SubscriptBox[\(Log\), \(10\)]\)(\!\(\*FractionBox[\(P\), \(N\)]\))","yaxisLabel"->None,"leg"-> leg,"colors"->colors,"plotRange"-> plotRange,"legPos"->legPos];
Return[{varIn,varNoise,varData,bias,tot,plot}];
];
{varIn,varNoise,varData,bias,tot}
   ];
