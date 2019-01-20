#include "TH1.h"
#include "TMath.h"
#include "TF1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "hist_fit.C"
#include "TAxis.h"
#include "TLatex.h"

TH1F* getRootHist(const char* histName) {
	
	return (TH1F*)gROOT->Get(histName);

}


Double_t calcSBRatio(TF1 fitFcn, Double_t energy) {
	
	// Define interval
	Double_t a = energy/2;
	Double_t b = energy*4/3;
	
	// Signal/background fcns
	TF1 *backFcn = new TF1("backFcn",background,TMath::Max(a,energy-1.5),TMath::Min(b,energy+1.5),2);
	
	TF1 *signalFcn = new TF1("signalFcn",gaussPeak,TMath::Max(a,energy-1.5),TMath::Min(b,energy+1.5),3);
	
	// writes the fit results into the par array
	Double_t par[5];
	
	fitFcn.GetParameters(par);
	
	backFcn->SetParameters(par);
	
	signalFcn->SetParameters(&par[2]);

	// Calc
	Double_t mean = fitFcn.GetParameter(3);
	Double_t var = fitFcn.GetParameter(4);
	Double_t std = TMath::Sqrt(var);
	
	Double_t signalIntegral = signalFcn->Integral(mean-std,mean+std,1e-12);
	Double_t backgroundIntegral = backFcn->Integral(mean-std,mean+std,1e-12);
	
	return signalIntegral / backgroundIntegral;

}	


TF1 * histFit2(const char *histRDName, const char *histADName, const char *histRL005Name,
			const char *histALName, const char *histABName, Double_t energy, const char * energyString, const char * canvasName, const char * canvasTitle) {
		
	return histFit(getRootHist(histRDName), getRootHist(histADName), getRootHist(histRL005Name),
					getRootHist(histALName), getRootHist(histABName), energy, energyString, canvasName, canvasTitle);
	
}


int LoopCases() {
	
	Int_t n = 6;
	
	Double_t sbRatioArrayRD[n], sbRatioArrayAD[n], sbRatioArrayRL005[n], sbRatioArrayAL[n], sbRatioArrayAB[n],
			energyArray[n], peakMeansRD[n], peakMeansAD[n], peakMeansRL005[n], peakMeansAL[n], peakMeansAB[n];
	
	for (int energyi = 0; energyi < n; energyi++){

		Double_t energy = 0.0;
		const char * energyString = NULL;
	
		switch (energyi) {
	
			case 0: energy = 1.0; energyString = "E1_0MeV"; break;
			case 1: energy = 1.5; energyString = "E1_5MeV"; break;
			case 2: energy = 2.0; energyString = "E2_0MeV"; break;
			case 3: energy = 2.5; energyString = "E2_5MeV"; break;
			case 4: energy = 3.5; energyString = "E3_5MeV"; break;
			case 5: energy = 5.0; energyString = "E5_0MeV"; break;
		
		}
		
		// Set all hist names
		char histRDName[64];
		sprintf(histRDName,"h_rel_loss_%s",energyString);
		char histADName[64];
		sprintf(histADName,"h_abs_rel_loss_%s",energyString);
		char histRL005Name[64];
		sprintf(histRL005Name,"h_relative_lambda0_05_%s",energyString);
		char histALName[64];
		sprintf(histALName,"h_abs_loss_%s",energyString);
		char histABName[64];
		sprintf(histABName,"h_1-neigh-highest_%s",energyString);
		
		// Energy string stuff
		char energyStringToSplit[strlen(energyString)+1];
		strncpy(energyStringToSplit, energyString, strlen(energyString)+1);
		const char* energyString1 = std::strtok(energyStringToSplit, "_");
		const char* energyString2 = std::strtok(nullptr, "_");
		
		char energyStringFinal[64];
		sprintf(energyStringFinal,"%s.%s",energyString1,energyString2);
		
		// Fit hists
		TF1 * fitFcns;
		fitFcns = histFit2(histRDName, histADName, histRL005Name, histALName, histABName,
							energy, energyStringFinal, energyStringFinal, energyStringFinal);
		
		// Extract fits
		TF1 fitFcnRD = *fitFcns;
		TF1 fitFcnAD = *(fitFcns+1);
		TF1 fitFcnRL005 = *(fitFcns+2);
		TF1 fitFcnAL = *(fitFcns+3);
		TF1 fitFcnAB = *(fitFcns+4);
		
		// Calc SB-ratios
		sbRatioArrayRD[energyi] = calcSBRatio(fitFcnRD,energy);
		sbRatioArrayAD[energyi] = calcSBRatio(fitFcnAD,energy);
		sbRatioArrayRL005[energyi] = calcSBRatio(fitFcnRL005,energy);
		sbRatioArrayAL[energyi] = calcSBRatio(fitFcnAL,energy);
		sbRatioArrayAB[energyi] = calcSBRatio(fitFcnAB,energy);
		
		// Extract peak means
		peakMeansRD[energyi] = fitFcnRD.GetParameter(3);
		peakMeansAD[energyi] = fitFcnAD.GetParameter(3);
		peakMeansRL005[energyi] = fitFcnRL005.GetParameter(3);
		peakMeansAL[energyi] = fitFcnAL.GetParameter(3);
		peakMeansAB[energyi] = fitFcnAB.GetParameter(3);
		
		// Current peak energy
		energyArray[energyi] = energy;
		
	}
	
	TCanvas * c2 = new TCanvas("c2","SB ratios",1500,100,900,600);
	c2->SetFillColor(0);
	c2->SetFrameFillColor(0);
	c2->SetLogy();
	
	// Make graphs
	TGraph * ratioGraphRD = new TGraph(n,peakMeansRD,sbRatioArrayRD);
	TGraph * ratioGraphAD = new TGraph(n,peakMeansAD,sbRatioArrayAD);
	TGraph * ratioGraphRL005 = new TGraph(n,peakMeansRL005,sbRatioArrayRL005);
	TGraph * ratioGraphAL = new TGraph(n,peakMeansAL,sbRatioArrayAL);
	TGraph * ratioGraphAB = new TGraph(n,peakMeansAB,sbRatioArrayAB);
	
	
	ratioGraphRD->SetTitle("Signal-bakgrund-f#ddot{o}rh#aallande");
	//ratioGraphRD->SetTitleFont(42);
	ratioGraphRD->GetXaxis()->SetTitle("Anpassad energi (MeV)");
	ratioGraphRD->GetYaxis()->SetTitle("S / B");
	ratioGraphRD->GetXaxis()->CenterTitle(true);
	ratioGraphRD->GetYaxis()->CenterTitle(true);
	ratioGraphRD->GetYaxis()->SetRangeUser(0.2,sbRatioArrayAB[5]+10);
	ratioGraphRD->GetXaxis()->SetTitleFont(42);
	ratioGraphRD->GetYaxis()->SetTitleFont(42);
	ratioGraphRD->GetXaxis()->SetTitleSize(0.045);
	ratioGraphRD->GetYaxis()->SetTitleSize(0.045);
	ratioGraphRD->GetXaxis()->SetLabelSize(0.045);
	ratioGraphRD->GetYaxis()->SetLabelSize(0.045);
	
	// Set marker colors, styles and sizes
	ratioGraphRD->SetMarkerColor(kBlue);
	ratioGraphAD->SetMarkerColor(kMagenta+2);
	ratioGraphRL005->SetMarkerColor(kGreen+1);
	ratioGraphAL->SetMarkerColor(kGreen+3);
	ratioGraphAB->SetMarkerColor(kRed);
	
	ratioGraphRD->SetMarkerStyle(3);
	ratioGraphAD->SetMarkerStyle(5);
	ratioGraphRL005->SetMarkerStyle(22);
	ratioGraphAL->SetMarkerStyle(23);
	ratioGraphAB->SetMarkerStyle(2);
	
	ratioGraphRD->SetMarkerSize(2);
	ratioGraphAD->SetMarkerSize(2);
	ratioGraphRL005->SetMarkerSize(2);
	ratioGraphAL->SetMarkerSize(2);
	ratioGraphAB->SetMarkerSize(2);
	
	// Set line colors, styles and widths
	ratioGraphRD->SetLineColor(kBlue);
	ratioGraphAD->SetLineColor(kMagenta+2);
	ratioGraphRL005->SetLineColor(kGreen+1);
	ratioGraphAL->SetLineColor(kGreen+3);
	ratioGraphAB->SetLineColor(kRed);
	
	ratioGraphRD->SetLineStyle(9);
	ratioGraphAD->SetLineStyle(10);
	ratioGraphRL005->SetLineStyle(2);
	ratioGraphAL->SetLineStyle(7);
	ratioGraphAB->SetLineStyle(1);
	
	ratioGraphRD->SetLineWidth(2);
	ratioGraphAD->SetLineWidth(2);
	ratioGraphRL005->SetLineWidth(2);
	ratioGraphAL->SetLineWidth(2);
	ratioGraphAB->SetLineWidth(2);
	
	// Draw graphs
	ratioGraphRD->Draw("APL");
	ratioGraphAD->Draw("PLsame");
	ratioGraphRL005->Draw("PLsame");
	ratioGraphAL->Draw("PLsame");
	ratioGraphAB->Draw("PLsame");
	
	TLegend *legend = new TLegend(0.15,0.85,0.5,0.6);
	//legend->SetHeader("Metod","C");
	legend->AddEntry(ratioGraphAB,"Addback","lp");
	legend->AddEntry(ratioGraphRD,"Relativ Doppler","lp");
	legend->AddEntry(ratioGraphAD,"Absolut Doppler","lp");
	legend->AddEntry(ratioGraphRL005,"Relativ #lambda = 0,05","lp");
	legend->AddEntry(ratioGraphAL,"Absolut E_{norm} = 5 MeV","lp");
	
	legend->Draw();

	
	return 0;
		
}	

	
