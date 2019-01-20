#!/bin/bash

BEAM_BETA=0.6:0.7
EVENTS=1000000

DATADIR=xb_sim/sim/

mkdir -p $DATADIR

for EGAMMA in 0.1 0.2 0.5 1.0 1.5 2.0 2.5 3.5 5.0
do
    ./land_geant4 \
	--xb=tree-gun-edep \
	--gun=dummy,beta=${BEAM_BETA},setboost,feed=f1 \
	--gun=from=f1:0.1,gamma,T=${EGAMMA}MeV,boost,isotropic \
	--gun=gamma,T=0.05:1MeV,prob=0.5,isotropic \
	--gun=gamma,T=0.05:2MeV,prob=0.5,isotropic \
	--tree=allevents,minentries=162,digi,${DATADIR}xb_line_E${EGAMMA}MeV.root \
	--events=${EVENTS} --np
done
