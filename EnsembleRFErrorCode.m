(* ::Package:: *)

(* ::Section:: *)
(*Ensembling Random Features*)


(* ::Text:: *)
(*This file contains the different functions for computing the generalisation error in random features. No changes should be made. Note that the functions to plot are given in a separate file named: PlotRandomFeatures.m*)
(*Please refer to HowTo.nb to see how to obtain the different curves. The file RandomFeaturesData.m contains the data (already computed) used for the figures in the paper.  *)
(*Throughout we define: \[Psi]1=P/D=(Number of features)/(Data dimension) and \[Psi]2=N/D=(Training set size)/(data dimension)*)


(* ::Subsubsection::Closed:: *)
(*Rules for Contractions*)


ClearAll[rules,mySim,getLnDet, getInv];
rules=Dispatch[{
id^a_:>id,
jj^a_/;a>1:>\[Epsilon]^(a-1) jj,
jj id:>jj,
id jj:>jj,
1/id:>id,
Times[id,jj,a___]:>Times[jj,a],
Times[jj,id,a___]:>Times[jj,a],
1/(a_ id +b_ jj):>getInv[a id +b jj,\[Epsilon]]
}];
mySim[A_]:=SeriesCoefficient[Expand[A],{id,0,1}]id + SeriesCoefficient[Expand[A],{jj,0,1}]jj;

getLnDet[R_]:=Module[{a,b},
a=SeriesCoefficient[R,{id,0,1}];
b=SeriesCoefficient[R,{jj,0,1}];
 \[Epsilon] Log[a] + Log[(1+b \[Epsilon] /a)]];
 
getInv[R_]:=Module[{aa,bb,a,b},
a=SeriesCoefficient[R,{id,0,1}];
b=SeriesCoefficient[R,{jj,0,1}];
bb=-b/(a^2+a b \[Epsilon])//FullSimplify;
aa=1/a;
aa id +bb  jj];


(* ::Subsubsection::Closed:: *)
(*Function for the Vanilla Action*)


Q=id q1+ jj q2;
R=r1 id +jj r2;

(*returns the action in the k=1 case*)
ClearAll[SVanilla];
SVanilla::usage = "SVanilla[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`] returns the value of the action, to zeroth order in \[Epsilon], in the vanila k=1 case";

SVanilla[\[Psi]1_,\[Psi]2_,\[Lambda]_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{S0,RQterm,LGL,A,QA,gW,lnDetgW,lnDetgX
},
RQterm=((R  getInv[Q]//Expand)//.rules//.{jj-> \[Epsilon], id-> \[Epsilon]});
RQterm=RQterm//Simplify;
LGL=((R* getInv[id+\[Mu]1^2 \[Psi]1 R]//Expand)//.rules)//mySim;
A=\[Mu]Star^2 id-( \[Mu]1^2 \[Mu]Star^2 \[Psi]1 LGL//Expand)//.rules//mySim;
QA=(Q A//Expand)//.rules//mySim;
gW=(id+ \[Psi]1 QA)//.rules//mySim;
lnDetgW= getLnDet[gW];
lnDetgX=getLnDet[(id+\[Mu]1^2 \[Psi]1 R//.rules)//mySim];

S0=\[Psi]2 lnDetgW +\[Psi]2 lnDetgX +( \[Psi]1^2 \[Psi]2 \[Lambda]) ((Q//.{id-> \[Epsilon], jj -> \[Epsilon]})//Simplify)+ RQterm +(1 -\[Psi]1) getLnDet[Q]- getLnDet[R]//Simplify;
S0=1/\[Epsilon] S0//Simplify;
S0=SeriesCoefficient[S0,{\[Epsilon],0,0}];
S0
];


(* ::Subsubsection::Closed:: *)
(*Function for the Mixed Action*)


QQ={{q1,q2},{q2,q1}};
RR={{r1,r2},{r2,r1}};
ClearAll[SMixed];
SMixed::usage = "SMixed[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`] returns the value of the action, to zeroth order in \[Epsilon], in the mixed case";
SMixed[\[Psi]1_,\[Psi]2_,\[Lambda]_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{gX,lnDetgW,lnDetgX,lnDetQ,lnDetR,cc,MX,A,QA,gW,RQinv,action},
(*GX*)
gX=IdentityMatrix[2] +\[Mu]1^2 \[Psi]1 RR;
lnDetgX=Log[Det[gX]]//Simplify;
cc=Inverse[gX];

(*MX*)
MX=(Dot[RR,cc]//Expand//Simplify);
A=(IdentityMatrix[2]-\[Mu]1^2 \[Psi]1 Expand[MX])\[Mu]Star^2//Expand//Simplify;
gW=(IdentityMatrix[2]+\[Psi]1  q1 A)//Expand//Simplify;
lnDetgW=Log[Det[gW]]//Simplify;

(*RRQ^-1*)
RQinv=Dot[RR,Inverse[QQ]]//Simplify;
RQinv=RR[[1,1]]/QQ[[1,1]]+RR[[2,2]]/QQ[[2,2]];

(*lnDetQ*)
lnDetQ=Log[Det[QQ]];

(*lnDetR*)
lnDetR=Log[Det[RR]];

(*action*)
action=(-\[Psi]1 lnDetQ + (Log[QQ[[1,1]]]+Log[QQ[[2,2]]])- lnDetR+ \[Psi]1^2 \[Psi]2 \[Lambda] Tr[QQ]+RQinv+\[Psi]2  lnDetgW + \[Psi]2 lnDetgX)//Expand//Simplify;
action
];


(* ::Subsubsection::Closed:: *)
(*Function for the Bagging Action*)


QQ={{q1,q2},{q2,q1}};
RR={{r1,r2},{r2,r1}};
ClearAll[SBag];
SBag::usage = "SBag[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`] returns the value of the action, to zeroth order in \[Epsilon], in the mixed case,with different data set";
SBag[\[Psi]1_,\[Psi]2_,\[Lambda]_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{gX,lnDetgW,lnDetgX,lnDetQ,lnDetR,cc,MX,A,QA,gW,RQinv,action},
(*GX*)
gX=IdentityMatrix[2] +\[Mu]1^2 \[Psi]1 {{RR[[1,1]],0},{0,RR[[1,1]]}};
lnDetgX=Log[Det[gX]]//Simplify;
cc=Inverse[gX];

(*MX*)
MX=(Dot[RR,cc]//Expand//Simplify);
A=(IdentityMatrix[2]-\[Mu]1^2 \[Psi]1 {{MX[[1,1]],0},{0,MX[[1,1]]}})\[Mu]Star^2//Expand//Simplify;
gW=(IdentityMatrix[2]+\[Psi]1 {{QQ[[1,1]]A[[1,1]],0},{0,QQ[[1,1]]A[[1,1]]}})//Expand//Simplify;
lnDetgW=Log[Det[gW]]//Simplify;

(*RRQ^-1*)
RQinv=Dot[RR,Inverse[QQ]]//Simplify;
RQinv=RR[[1,1]]/QQ[[1,1]]+RR[[2,2]]/QQ[[2,2]];

(*lnDetQ*)
lnDetQ=Log[Det[QQ]];

(*lnDetR*)
lnDetR=Log[Det[RR]];

(*action*)
action=(-\[Psi]1 lnDetQ + (Log[QQ[[1,1]]]+Log[QQ[[2,2]]])- lnDetR+ \[Psi]1^2 \[Psi]2 \[Lambda] Tr[QQ]+RQinv+\[Psi]2  lnDetgW + \[Psi]2 lnDetgX)//Expand//Simplify;
action
];


(* ::Subsubsection::Closed:: *)
(*Auxiliary Functions*)


(*ClearAll[getRes];
getRes::usage = "getRes[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`] returns the solution of the fixed point equation for the order parameters q0 and r0"
getRes[\[Psi]1_,\[Psi]2_,\[Lambda]V_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{h,res,nres},
h[q1_,r1_]:=SeriesCoefficient[SeriesCoefficient[SVanilla[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star],{q2,0,0}],{r2,0,0}];
	res=NSolve[{D[h[q1,r1],r1]==0,D[h[q1,r1],q1]==0},{q1,r1}];
	nres=Select[res,(Im[#[[1]][[2]]]==0&&Im[#[[2]][[2]]]==0&&#[[1]][[2]]>0&&#[[2]][[2]]>0)&];
	If[Length[nres]>=1,nres=nres[[1]],nres=Nothing];
	nres
];*)
(*newest*)
ClearAll[getRes];
getRes::usage = "getRes[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`] returns the solution of the fixed point equation for the order parameters q0 and r0";
getRes[\[Psi]1_,\[Psi]2_,\[Lambda]V_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{h,res,nres,S0},
S0=SVanilla[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star]//.{r2->0,q2->0};
	res=NSolve[{D[S0,r1]==0,D[S0,q1]==0},{q1,r1}];
	nres=Select[res,(Im[#[[1]][[2]]]==0&&Im[#[[2]][[2]]]==0&&#[[1]][[2]]>0&&#[[2]][[2]]>0)&];
	If[Length[nres]>=1,nres=nres[[1]],nres=Nothing];
	nres
];

ClearAll[getValue];
getValue::usage = "getValue[A,numres,hqq,hrq,hrr] replaces, in expressions A, the oscilation around the minima by their relative value by the value of the hessian evaluated at the fix point solution";
getValue[A_,numres_,hqq_,hrq_,hrr_]:=Module[{AA},
AA=A//.Dispatch[numres];
AA=Series[AA,{r2,0,2},{q2,0,2}];
AA=Normal[AA];
AA=(AA//Expand)//.Dispatch[{Times[c___,q2,r2^b_ ]/;(b>1 ):> 0.0,Times[c___,q2^b_,r2]/;(b>1 ):> 0.0 ,Times[c___,r2^aa_,q2^b_]/;(b>1 Or aa>1 ):> 0.0}];
AA=(AA//Expand)//. Dispatch[{Times[c___,r2,r2]:>c r2r2,Times[c___,q2,r2]:>c q2r2,Times[c___,q2,q2]:>c  q2q2}]//Simplify;
AA=(AA//Expand)//.Dispatch[{Times[c___,r2]:>0,Times[c___,q2]:>0}];
AA=AA//.Dispatch[{r2r2 -> hrr, q2r2 -> hrq, q2q2 -> hqq}];
AA=AA//.Dispatch[numres];
AA
];


(* ::Subsubsection::Closed:: *)
(*Function for the Vanilla Error*)


ClearAll[getHessInvVanilla];
getHessInvVanilla[numres_,\[Psi]1_,\[Psi]2_,\[Lambda]V_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{hessInv,res},
hessInv=Table[Table[(D[D[SVanilla[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star],i],j]//.Dispatch[Join[numres,{q2->0,r2->0}]])//Simplify,{i,{q1,r1,q2,r2}}],{j,{q1,r1,q2,r2}}];
	hessInv=hessInv//.Dispatch[{q2->0,r2->0}];
	hessInv=hessInv//Simplify;
	hessInv=Inverse[hessInv];
	hessInv=hessInv//.{\[Epsilon]->0};
	hessInv
];


getVanillaTerms::usage = "getVanillaTerms is a function that takes 5 parameters 2 of which have a default value:
getVanillaTerms[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1_:0.5,\[Mu]Star]
returns the vanilla terms in the error {\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3}";
ClearAll[getVanillaTerms];
getVanillaTerms[\[Psi]1_,\[Psi]2_,\[Lambda]V_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{aa,bb,MW,MX,A,QA,gW,
MX1,MX2,A1,A2,QA1,QA2,Q1,Q2,u,u2,MW1,MW2,MM1,MM2,MMM1,MMM2,hqq,hqr,hrr,HH,getPsi3,getPsi2,getPsi1,NX1,NX2,numres
 },
numres=getRes[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star];
If[numres==Nothing, Return[{0,0,0}] ];

HH=getHessInvVanilla[numres,\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star];
{hqq,hqr,hrr}={HH[[3]][[3]],HH[[3]][[4]],HH[[4]][[4]]};

aa=1/(1+\[Mu]1^2 \[Psi]1 r1)//.numres//Simplify;
bb=-\[Mu]1^2 \[Psi]1 r2/((1+\[Mu]1^2 \[Psi]1 r1)(1+\[Mu]1^2 \[Psi]1 r1))//.numres//Simplify;
MX1=(r1*aa//.numres)//Simplify;
MX2=(r1 bb + r2 aa)//.numres//Simplify;

A1=(\[Mu]Star^2-\[Mu]Star^2 * \[Mu]1^2 \[Psi]1 MX1)//.numres//Simplify;
A2 =-(\[Mu]Star^2 \[Mu]1^2 \[Psi]1 MX2)//.numres//Simplify;

Q1=numres[[1]][[2]]//Simplify;
Q2=q2;
QA1=A1 Q1;
QA2= A1 Q2 + Q1 A2//Simplify;

u=(1+\[Psi]1 QA1)//.numres//Simplify;
u2= (\[Psi]1 QA2)//.numres//Simplify;
MW1=Q1/u//Simplify;
MW2=(-Q1 u2)/(u*u)+Q2/u//Simplify;

MM1=MX1 MW1 //Expand//Simplify;
MM2= MX1 MW2 + MW1 MX2//Expand//Simplify;
MMM1=MM1 MX1//Expand//Simplify;
MMM2=MM2 MX1+MM1 MX2//Expand//Simplify;

NX1=aa MX1//Expand//Simplify;
NX2=aa MX2 + bb MX1//Expand//Simplify;

getPsi3:=Module[{t1,t2,t3,t4,psi3},
t1=( \[Psi]2 \[Psi]1^2 (\[Mu]1^2 r2 + \[Mu]Star^2 q2) \[Mu]1^2 MX2)//Expand//Simplify;
t2=((\[Psi]2 \[Psi]1^2 (\[Mu]1^2 r2 + \[Mu]Star^2 q2) )\[Mu]Star^2 MW2)//Expand//Simplify;
t3=(\[Psi]2 \[Psi]1^2 (\[Mu]1^2 r2 + \[Mu]Star^2 q2)) \[Mu]1^4 \[Mu]Star^2  \[Psi]1^2 (2 MX1 MX2 MW1 + MX1^2 MW2)//Expand//Simplify;
t4=-(\[Psi]2 \[Psi]1^2 (\[Mu]1^2 r2 + \[Mu]Star^2 q2)) \[Mu]1^2 \[Psi]1 \[Mu]Star^2 2 (MX1 MW2+ MW1 MX2)//Expand//Simplify;
t1=getValue[t1,numres,hqq,hqr,hrr];
t2=getValue[t2,numres,hqq,hqr,hrr];
t3=getValue[t3,numres,hqq,hqr,hrr];
t4=getValue[t4,numres,hqq,hqr,hrr];
psi3 = -t1 - t2 - t3 - t4;
psi3
];

getPsi1:=Module[{t1,t2,t3,psi1},
t1=(\[Mu]1^2 \[Psi]1 \[Psi]2 (MX1+MX2))//Expand//Simplify;
t2=(\[Mu]1^2 \[Psi]1 \[Psi]2)( \[Mu]1^2 \[Psi]1^2 \[Mu]Star^2 (MMM1+MMM2))//Expand//Simplify;
t3=(\[Mu]1^2 \[Psi]1 \[Psi]2)(  \[Psi]1 \[Mu]Star^2 (MM1+MM2 ))//Expand//Simplify;
t1=getValue[t1,numres,hqq,hqr,hrr];
t2=getValue[t2,numres,hqq,hqr,hrr];
t3=getValue[t3,numres,hqq,hqr,hrr];
psi1 = t1 + t2 - t3 ;
psi1
];

getPsi2:=Module[{t1,t2,t3,psi2},
t1=NX2+1/\[Psi]2  MX2+ 2*(\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2)(MW2 MX1 NX1+MW1 MX2 NX1+MW1 MX1 NX2) + (\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2)/\[Psi]2 (MW2 MX1 MX1+MW1 MX2 MX1+MW1 MX1 MX2)+(\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2)^2 (2 MW1 MW2 MX1^2 NX1+2 MW1^2 MX1 MX2 NX1+MW1^2 MX1^2 NX2)//Expand//Simplify;
t1=\[Psi]1^2 \[Psi]2^2 (\[Mu]Star^2 q2+\[Mu]1^2 r2)(\[Mu]1^2) t1//Expand//Simplify;
t2=(NX1 MW2 + NX2 MW1)+1/\[Psi]2 (MX1 MW2 + MX2 MW1)+ (\[Mu]1 \[Mu]Star \[Psi]1)^2 (2 MW1 MW2 MX1 NX1+MW1^2 MX2 NX1+MW1^2 MX1 NX2);
t2=2*\[Psi]1^2 \[Psi]2^2 (\[Mu]Star^2 q2+\[Mu]1^2 r2)(\[Mu]1^2 \[Psi]1 \[Mu]Star^2)t2 //Expand//Simplify;
t3=1/\[Psi]2 MW2 + (2 MW1 NX1 MW2 + MW1^2 NX2  )(\[Mu]1 \[Mu]Star \[Psi]1)^2;
t3=\[Psi]2^2 \[Psi]1^2  (\[Mu]Star^2 q2+\[Mu]1^2 r2)\[Mu]Star^2  t3 //Expand//Simplify;
t1=getValue[t1,numres,hqq,hqr,hrr];
t2=getValue[t2,numres,hqq,hqr,hrr];
t3=getValue[t3,numres,hqq,hqr,hrr];
psi2 =- t1 + t2 - t3 ;
psi2
];
{getPsi1,getPsi2,getPsi3}
];


ErrorVanilla::usage = "ErrorVanilla is a function that takes 9 parameters 6 of which have a default value:
ErrorVanilla[\[Psi]1,\[Psi]2,\[Lambda],F1_:1.0,FStar_:0.1,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star]
 returns 4 parameters: {\[Lambda] value, \[Psi]1 value, \[Psi]2 value, generalisation for vanilla}";
ClearAll[ErrorVanilla];
ErrorVanilla[\[Psi]1_,\[Psi]2_,\[Lambda]_,F1_:1.0,FStar_:0.0,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{\[Lambda]V=\[Lambda],error,\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3},
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3}=getVanillaTerms[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
error=F1^2 (1-2 \[CapitalPsi]1+ \[CapitalPsi]2)+(FStar^2+\[Tau]^2) \[CapitalPsi]3 + FStar^2;
{\[Lambda], \[Psi]1, \[Psi]2,error}
];


(* ::Subsubsection:: *)
(*Function for the Mixed Error*)


ClearAll[remplace];
remplace=Dispatch[{r2->0.0,q2->0.0}];

ClearAll[getHessInvMixed];
getHessInvMixed[numres_,\[Psi]1_,\[Psi]2_,\[Lambda]M_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{hessInv,res},
hessInv=Table[Table[(D[D[SMixed[\[Psi]1,\[Psi]2,\[Lambda]M],i],j]//.remplace)//.numres//Simplify,{i,{q1,r1,q2,r2}}],{j,{q1,r1,q2,r2}}];
hessInv=Inverse[hessInv];
hessInv
];

getMixedTerms::usage = "getMixedTerms is a function that takes 5 parameters 2 of which have a default value:
getMixedTerms[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1_:0.5,\[Mu]Star]
returns the vanilla terms in the error {\[CapitalPsi]4,\[CapitalPsi]5}";
ClearAll[getMixedTerms];
getMixedTerms[\[Psi]1_,\[Psi]2_,\[Lambda]M_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{numres,HH,getPsi4,getPsi5,gX,cc,MX,A,MW,MXMW,MXMWMX,NX,MXMWNX,MXMWNXMWMX,NXMW,MXMWNXMW,MWNXMW,hqq,hrq,hrr},
(*solution to fix point equations*)
numres=getRes[\[Psi]1,\[Psi]2,\[Lambda]M,\[Mu]1,\[Mu]Star];
If[numres==Nothing, Return[{0,0}]];
HH=getHessInvMixed[numres,\[Psi]1,\[Psi]2,\[Lambda]M,\[Mu]1,\[Mu]Star];
{hqq,hrq,hrr}={2* HH[[3,3]],2*HH[[3,4]],2*HH[[4,4]]};

gX=IdentityMatrix[2]+\[Mu]1^2 \[Psi]1 RR;
cc=Inverse[gX]//Simplify;
MX=Dot[RR,(Inverse[gX]//Simplify)]//Simplify;
A=(IdentityMatrix[2]-\[Mu]1^2 \[Psi]1 Expand[MX])\[Mu]Star^2//Expand//Simplify;
MW=Inverse[IdentityMatrix[2]+\[Psi]1  q1 A]//Simplify;
MW=q1 IdentityMatrix[2]-q1^2 \[Psi]1 Dot[A,MW]//Simplify;

MXMW=(Dot[MX,MW]//Expand);

MXMWMX=(Dot[MXMW,MX]//Expand);

NX=(Dot[cc,MX]//Expand);
MXMWNX=(Dot[MXMW,NX]//Expand);
MXMWNXMWMX=(Dot[MXMWNX,Dot[MW,MX]]//Expand);
NXMW=(Dot[NX,MW]//Expand);
MXMWNXMW=(Dot[MXMW,NXMW]//Expand);
MWNXMW=(Dot[MW,NXMW]);

getPsi4:=Module[{t1,t2,t3,psi4},
t1= NX[[1,2]]+1/\[Psi]2 MX[[1,2]]+ 2*(\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2) MXMWNX[[1,2]]+(\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2)/\[Psi]2 MXMWMX[[1,2]]+(\[Mu]1^2 \[Mu]Star^2 \[Psi]1^2)^2 MXMWNXMWMX[[1,2]];
t1=(\[Mu]1^2 \[Psi]1^2 \[Psi]2^2 )(\[Mu]1^2 r2) t1//Expand;

t2=NXMW[[1,2]]+1/\[Psi]2 MXMW[[1,2]]+ (\[Mu]1 \[Mu]Star \[Psi]1)^2 MXMWNXMW[[1,2]];
t2=2*(\[Mu]1^2 \[Psi]2^2 \[Psi]1^3 \[Mu]Star^2)( r2 \[Mu]1^2)t2 //Expand;

t3=1/\[Psi]2 MW[[1,2]] + MWNXMW[[1,2]](\[Mu]1 \[Mu]Star \[Psi]1)^2 ;
t3=\[Psi]2^2 \[Psi]1^2  \[Mu]Star^2 (\[Mu]1^2 r2) t3 //Expand;
t1=getValue[t1,numres,hqq,hrq,hrr];
t2=getValue[t2,numres,hqq,hrq,hrr];
t3=getValue[t3,numres,hqq,hrq,hrr];
psi4=t1-t2+t3//Simplify;
psi4
];

getPsi5:=Module[{t1,t2,t3,t4,psi5},
t1=\[Mu]1^2 \[Psi]1^2 \[Psi]2 r2 \[Mu]1^2 MX[[1,2]]//Expand;
t2=(\[Mu]1^2 \[Psi]1^2 \[Psi]2 r2) \[Mu]1^4 \[Mu]Star^2 \[Psi]1^2 MXMWMX[[1,2]]//Expand;
t3=(\[Mu]1^2 \[Psi]1^2 \[Psi]2 r2) \[Mu]1^2 \[Mu]Star^2 \[Psi]1 (MXMW[[1,2]]+MXMW[[2,1]]);
t4=\[Mu]1^2 \[Psi]1^2 \[Psi]2 r2 \[Mu]Star^2 MW[[1,2]]//Expand;
t1=getValue[t1,numres,hqq,hrq,hrr];
t2=getValue[t2,numres,hqq,hrq,hrr];
t3=getValue[t3,numres,hqq,hrq,hrr];
t4=getValue[t4,numres,hqq,hrq,hrr];
psi5=(t1+t2-t3+t4)//Simplify;
psi5
];
{getPsi4,getPsi5}
];

ClearAll[ErrorMixed];
ErrorMixed::usage = "ErrorMixed is a function that takes 9 parameters 6 of which have a default value:
ErrorMixed[\[Psi]1,\[Psi]2,\[Lambda],k_:3.0,F1_:1.0,FStar_:0.1,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star]
 returns 6 outputs: {\[Lambda] ,\[Psi]1,\[Psi]2,k, generalisation error vanilla, generalisation error ensembled k, k\[Rule]\[Infinity]: kinf}";
Options[ErrorMixed]:={"pts"->{0}};
ErrorMixed[\[Psi]1_,\[Psi]2_,\[Lambda]_,k_:3.0,F1_:1.0,FStar_:0.1,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`,OptionsPattern[]]:=Module[{errVanilla,errMixed,errKinf,\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5,\[Lambda]V=\[Lambda],\[Lambda]M=\[Lambda],
pts=OptionValue["pts"]},

If[Length[pts]==1,
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3}=getVanillaTerms[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star];
Print["All done Vanilla"];
{\[CapitalPsi]4,\[CapitalPsi]5}=getMixedTerms[\[Psi]1,\[Psi]2,\[Lambda]V,\[Mu]1,\[Mu]Star];
Print["All done Mixed"];,
{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3,\[CapitalPsi]4,\[CapitalPsi]5}=#[[2]]&/@pts[[1;;5]];
];
If[{\[CapitalPsi]1,\[CapitalPsi]2,\[CapitalPsi]3}=={0,0,0},
errVanilla=0,
errVanilla=F1^2 (1-2 \[CapitalPsi]1+ \[CapitalPsi]2)+(FStar^2+\[Tau]^2) \[CapitalPsi]3 + FStar^2];

If[
{\[CapitalPsi]4,\[CapitalPsi]5}=={0,0},
errMixed=0;errKinf=0;,
errMixed=F1^2 (1-2 \[CapitalPsi]1+ 1/k \[CapitalPsi]2)+1/k (FStar^2+\[Tau]^2) \[CapitalPsi]3 + FStar^2+ (1-1/k) (F1^2 \[CapitalPsi]4+(FStar^2+\[Tau]^2)\[CapitalPsi]5);
errKinf=F1^2 (1-2 \[CapitalPsi]1)+ FStar^2+ (F1^2 \[CapitalPsi]4+(FStar^2+\[Tau]^2)\[CapitalPsi]5);
];
{\[Lambda],\[Psi]1,\[Psi]2,k,errVanilla,errMixed,errKinf}
];


Length[{0}]


(* ::Subsubsection::Closed:: *)
(*Function for \[CapitalPsi]4*)


ClearAll[remplace];
remplace=Dispatch[{r2->0.0,q2->0.0}];

ClearAll[getHessInvBag];
getHessInvBag[numres_,\[Psi]1_,\[Psi]2_,\[Lambda]M_,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{hessInv,res},
hessInv=Table[Table[(D[D[SBag[\[Psi]1,\[Psi]2,\[Lambda]M],i],j]//.remplace)//.numres//Simplify,{i,{q1,r1,q2,r2}}],{j,{q1,r1,q2,r2}}];
hessInv=Inverse[hessInv];
hessInv
];


ClearAll[getPsi4];
getPsi4::usage = "getPsi4 is a function that takes 8 parameters 5 of which have a default value:
ErrorMixed[\[Psi]1,\[Psi]2,\[Lambda],F1_:1.0,FStar_:0.1,\[Tau]_:0.0,\[Mu]1_:0.5,\[Mu]Star]
 returns 1 outputs: \[CapitalPsi]4 which is the additional term appearing in the bias and in the data variance";
ClearAll[getPsi4];
getPsi4[\[Psi]1_,\[Psi]2_,\[Lambda]_,F1_:1.0,FStar_:0.0,\[Mu]1_:0.5,\[Mu]Star_:0.3014051374945435`]:=Module[{psi4,gX,cc,MX,MW,MA,MXMW,MXMWMX,A,MM1,MM2,MW1,MW2,t1,t2,t3,t4,numres,hess,hqq,hrq,hrr,a,b,NX},
(*solution to fix point equations*)

numres=getRes[\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
If[numres==Nothing,Return[{\[Lambda],0,0,0}]];
(*Print[numres];*)
(*obtaining the hessian*)
hess=getHessInvBag[numres,\[Psi]1,\[Psi]2,\[Lambda],\[Mu]1,\[Mu]Star];
{hqq,hrq,hrr}={hess[[3,3]]*2,hess[[3,4]]*2,hess[[4,4]]*2};
gX=IdentityMatrix[2]+\[Mu]1^2 \[Psi]1 {{RR[[1,1]],0},{0,RR[[2,2]]}}//.numres;
cc=Inverse[gX]//Simplify;
MX=Dot[{{RR[[1,1]],0},{0,RR[[2,2]]}},cc]//.numres;
A={{\[Mu]Star^2-\[Mu]Star^2 \[Psi]1 \[Mu]1^2 MX[[1,1]],0},{0,\[Mu]Star^2-\[Mu]Star^2 \[Psi]1 \[Mu]1^2 MX[[2,2]]}}//Expand//Simplify;
MW={{QQ[[1,1]]/(1+\[Psi]1 A[[1,1]]QQ[[1,1]]),0},{0,QQ[[2,2]]/(1+\[Psi]1 A[[2,2]]QQ[[2,2]])}}//.numres//Simplify;

NX=1/(1+\[Mu]1^2 \[Psi]1 RR[[1,1]]) r2 1/(1+\[Mu]1^2 \[Psi]1 RR[[2,2]])//.numres;
a=(\[Mu]1 \[Mu]Star \[Psi]1);
b=\[Mu]1^2 \[Psi]1 \[Psi]2^2;

t1=NX+2*a^2*(NX MW[[2,2]] MX[[2,2]])+a^4 MX[[1,1]]MW[[1,1]]NX MW[[2,2]] MX[[2,2]];

t1=b r2 \[Mu]1^2 \[Psi]1 t1;


t2=NX MW[[1,1]]+ (\[Mu]1 \[Mu]Star \[Psi]1)^2 MX[[2,2]] MW[[2,2]] NX MW[[1,1]];
t2=2*(\[Mu]1^2 \[Psi]2^2 \[Psi]1^3 \[Mu]Star^2)( r2 \[Mu]1^2)t2 //Expand//Simplify;

t3=MW[[2,2]]NX MW[[1,1]](\[Mu]1 \[Mu]Star \[Psi]1)^2 \[Psi]2;
t3=\[Psi]2 \[Psi]1^2  \[Mu]Star^2 (\[Mu]1^2 r2) t3 //Expand//Simplify;
t1=getValue[t1,numres,hqq,hrq,hrr];
t2=getValue[t2,numres,hqq,hrq,hrr];
t3=getValue[t3,numres,hqq,hrq,hrr];
t1-t2+t3
];
