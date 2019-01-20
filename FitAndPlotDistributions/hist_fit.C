#include "TH1.h"
#include "TMath.h"
#include "TF1.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TAxis.h"

Double_t background(Double_t *x, Double_t *par) {
	// par[0]: amplitude
	// par[1]: slope
	return par[0]*TMath::Exp(-par[1]*x[0]);
	// return par[0]*x[0] + par[1];
	// return par[0]/x[0];
}

Double_t gaussPeak(Double_t *x, Double_t *par) {
	// par[0]: amplitude
	// par[1]: mean
	// par[2]: variance
	return par[0]*TMath::Exp(-(x[0]-par[1])*(x[0]-par[1])/(2*par[2]));
}

Double_t fitFunction(Double_t *x, Double_t *par) {
	return background(x,par) + gaussPeak(x,&par[2]);
}

TF1 * histFit(TH1F *histRD, TH1F *histAD, TH1F *histRL005, TH1F *histAL, TH1F *histAB,
			Double_t energy, const char * energyString, const char * canvasName, const char * canvasTitle) {
	
	
	TCanvas *c = new TCanvas(canvasName,canvasTitle,energy*100,energy*100,810,600);
	c->SetFillColor(0);
	c->SetFrameFillColor(0);
	//c->SetGrid();
	
	// Define interval
	Double_t a = energy/2;
	Double_t b = energy*4/3;
	
	// Set fitfcns and plot params
	Double_t x0fit = TMath::Max(a,energy - 2.5);
	Double_t x1fit = TMath::Min(b,energy + 1.5);
	
	TF1 *fitFcnRD = new TF1("fitFcnRD",fitFunction,x0fit,x1fit,5);
	fitFcnRD->SetLineWidth(2);
	fitFcnRD->SetLineColor(kBlue);
	
	TF1 *fitFcnAD = new TF1("fitFcnAD",fitFunction,x0fit,x1fit,5);
	fitFcnAD->SetLineWidth(2);
	fitFcnAD->SetLineColor(kBlack);
	
	TF1 *fitFcnRL005 = new TF1("fitFcnRL005",fitFunction,x0fit,x1fit,5);
	fitFcnRL005->SetLineWidth(2);
	fitFcnRL005->SetLineColor(kBlack);
	
	TF1 *fitFcnAL = new TF1("fitFcnAL",fitFunction,x0fit,x1fit,5);
	fitFcnAL->SetLineWidth(2);
	fitFcnAL->SetLineColor(kBlack);
	
	TF1 *fitFcnAB = new TF1("fitFcnAB",fitFunction,x0fit,x1fit,5);
	fitFcnAB->SetLineWidth(2);
	fitFcnAB->SetLineColor(kRed);
		
	// Set params start values
	Double_t expAmp0 = 1e4*energy;
	Double_t expSlope0 = 2.0;
	Double_t gaussAmp0 = 2e3/energy;
	Double_t gaussMean0 = energy;
	Double_t gaussVar0 = 0.1;
	
	fitFcnRD->SetParameters(expAmp0,expSlope0,gaussAmp0,gaussMean0,gaussVar0);
	fitFcnAD->SetParameters(expAmp0,expSlope0,gaussAmp0,gaussMean0,gaussVar0);
	fitFcnRL005->SetParameters(expAmp0,expSlope0,gaussAmp0,gaussMean0,gaussVar0);
	fitFcnAL->SetParameters(expAmp0,expSlope0,gaussAmp0,gaussMean0,gaussVar0);
	fitFcnAB->SetParameters(expAmp0,expSlope0,gaussAmp0,gaussMean0,gaussVar0);
	
	// Set params limits
	fitFcnRD->SetParLimits(0,1e3,1e5);
	fitFcnRD->SetParLimits(1,0,10);
	fitFcnRD->SetParLimits(2,10,1e4);
	fitFcnRD->SetParLimits(3,a,b);
	fitFcnRD->SetParLimits(4,0,1);
	
	fitFcnAD->SetParLimits(0,1e3,1e5);
	fitFcnAD->SetParLimits(1,0,10);
	fitFcnAD->SetParLimits(2,10,1e4);
	fitFcnAD->SetParLimits(3,a,b);
	fitFcnAD->SetParLimits(4,0,1);
	
	fitFcnRL005->SetParLimits(0,1e3,1e5);
	fitFcnRL005->SetParLimits(1,0,10);
	fitFcnRL005->SetParLimits(2,10,1e4);
	fitFcnRL005->SetParLimits(3,a,b);
	fitFcnRL005->SetParLimits(4,0,1);
	
	fitFcnAL->SetParLimits(0,1e3,1e5);
	fitFcnAL->SetParLimits(1,0,10);
	fitFcnAL->SetParLimits(2,10,1e4);
	fitFcnAL->SetParLimits(3,a,b);
	fitFcnAL->SetParLimits(4,0,1);
	
	fitFcnAB->SetParLimits(0,1e3,1e5);
	fitFcnAB->SetParLimits(1,0,10);
	fitFcnAB->SetParLimits(2,10,1e4);
	fitFcnAB->SetParLimits(3,a,b);
	fitFcnAB->SetParLimits(4,0,1);	

	
	// Set params names
	const char *backAmpRD = "backAmpRD";
	const char *backSlopeRD = "backSlopeRD";
	const char *gaussAmpRD = "gaussAmpRD";
	const char *gaussMeanRD = "gaussMeanRD";
	const char *gaussVarRD = "gaussVarRD";
	
	const char *backAmpAD = "backAmpAD";
	const char *backSlopeAD = "backSlopeAD";
	const char *gaussAmpAD = "gaussAmpAD";
	const char *gaussMeanAD = "gaussMeanAD";
	const char *gaussVarAD = "gaussVarAD";
	
	const char *backAmpRL005 = "backAmpRL005";
	const char *backSlopeRL005 = "backSlopeRL005";
	const char *gaussAmpRL005 = "gaussAmpRL005";
	const char *gaussMeanRL005 = "gaussMeanRL005";
	const char *gaussVarRL005 = "gaussVarRL005";
	
	const char *backAmpAL = "backAmpAL";
	const char *backSlopeAL = "backSlopeAL";
	const char *gaussAmpAL = "gaussAmpAL";
	const char *gaussMeanAL = "gaussMeanAL";
	const char *gaussVarAL = "gaussVarAL";
	
	const char *backAmpAB = "backAmpAB";
	const char *backSlopeAB = "backSlopeAB";
	const char *gaussAmpAB = "gaussAmpAB";
	const char *gaussMeanAB = "gaussMeanAB";
	const char *gaussVarAB = "gaussVarAB";
	
	fitFcnRD->SetParNames(backAmpRD,backSlopeRD,gaussAmpRD,gaussMeanRD,gaussVarRD);
	fitFcnAD->SetParNames(backAmpAD,backSlopeAD,gaussAmpAD,gaussMeanAD,gaussVarAD);
	fitFcnRL005->SetParNames(backAmpRL005,backSlopeRL005,gaussAmpRL005,gaussMeanRL005,gaussVarRL005);
	fitFcnAL->SetParNames(backAmpAL,backSlopeAL,gaussAmpAL,gaussMeanAL,gaussVarAL);
	fitFcnAB->SetParNames(backAmpAB,backSlopeAB,gaussAmpAB,gaussMeanAB,gaussVarAB);
	
	histRD->SetLineColor(kBlue);
	histAB->SetLineColor(kRed);
	histRD->SetLineWidth(2);
	histAB->SetLineWidth(2);
	histRD->SetMarkerStyle(20);
	histAB->SetMarkerStyle(24);
	histRD->SetMarkerSize(1);
	histAB->SetMarkerSize(1);
	histRD->SetMarkerColor(kBlue-9);
	histAB->SetMarkerColor(kRed-9);
	
	
	
	// Hist titles
	char title[64];
	sprintf(title,"%s","Rekonstruerat spektrum med #gamma-topp vid 2,5 MeV");
	histRD->SetTitle(title);
	histAB->SetTitle("AB rekonstruktion");
	histAL->SetTitle("AL rekonstruktion");
	histRL005->SetTitle("RL005 rekonstruktion");
	histAD->SetTitle("AD rekonstruktion");
	// Axis
	histRD->GetXaxis()->SetTitle("Energy (MeV)");
	histRD->GetYaxis()->SetTitle("Counts / 10 keV");
	
	histRD->GetXaxis()->SetTitleFont(42);
	histRD->GetYaxis()->SetTitleFont(42);
	histRD->GetXaxis()->SetTitleSize(0.045);
	histRD->GetYaxis()->SetTitleSize(0.045);
	histRD->GetXaxis()->SetLabelSize(0.045);
	histRD->GetYaxis()->SetLabelSize(0.045);
	histRD->GetYaxis()->SetTitleOffset(1.25);
	
	
	
	// Fit
	histRD->Fit("fitFcnRD","RM","p");
	histAD->Fit("fitFcnAD","RM0","same"); 	
	histRL005->Fit("fitFcnRL005","RM0","same"); 	
	histAL->Fit("fitFcnAL","RM0","same"); 	
	histAB->Fit("fitFcnAB","RM","psame"); 	
	
	// Set axis
	histRD->GetXaxis()->SetRangeUser(x0fit,x1fit);
	histRD->GetYaxis()->SetRangeUser(-100,3600);
	histRD->GetXaxis()->CenterTitle(true);
	histRD->GetYaxis()->CenterTitle(true);
	
	
	//histAB->GetXaxis()->SetRangeUser(TMath::Max(a,energy-1),TMath::Min(b,energy+1));
	//histAB->GetYaxis()->SetRangeUser(-100,9000/energy);
	
	// Signal/background fcns
	TF1 *backFcnRD = new TF1("backFcnRD",background,x0fit,x1fit,2);
	backFcnRD->SetTitle("RD bakgrundsanpassning");
	backFcnRD->SetLineColor(kBlue+2);
	backFcnRD->SetLineWidth(2);
	backFcnRD->SetLineStyle(2);
	
	TF1 *backFcnAB = new TF1("backFcnAB",background,x0fit,x1fit,2);
	backFcnAB->SetTitle("AB bakgrundsanpassning");
	backFcnAB->SetLineColor(kRed+2);
	backFcnAB->SetLineWidth(2);
	backFcnAB->SetLineStyle(2);
	
	TF1 *signalFcnRD = new TF1("signalFcnRD",gaussPeak,x0fit,x1fit,3);
	signalFcnRD->SetTitle("RD signalanpassning");
	signalFcnRD->SetLineColor(kBlue+2);
	signalFcnRD->SetLineWidth(2);
	signalFcnRD->SetLineStyle(9);
	
	TF1 *signalFcnAB = new TF1("signalFcnAB",gaussPeak,x0fit,x1fit,3);
	signalFcnAB->SetTitle("AB signalanpassning");
	signalFcnAB->SetLineColor(kRed+2);
	signalFcnAB->SetLineWidth(2);
	signalFcnAB->SetLineStyle(9);
	
	// writes the fit results into the par array
	Double_t parRD[5];
	Double_t parAB[5];
	
	fitFcnRD->GetParameters(parRD);
	fitFcnAB->GetParameters(parAB);
	
	backFcnRD->SetParameters(parRD);
	backFcnAB->SetParameters(parAB);
	backFcnRD->Draw("same");
	backFcnAB->Draw("same");
	
	signalFcnRD->SetParameters(&parRD[2]);
	signalFcnAB->SetParameters(&parAB[2]);
	signalFcnRD->Draw("same");
	signalFcnAB->Draw("same");
	
	TLegend *legend = new TLegend(0.225,0.64,0.89,0.88);
	legend->SetNColumns(2);
	legend->AddEntry(histRD,"Neural network","p");
	legend->AddEntry(histAB,"Conventional method","p");
	legend->AddEntry(fitFcnRD,"Fit (S+B)","l");
	legend->AddEntry(fitFcnAB,"Fit (S+B)","l");
	legend->AddEntry(signalFcnRD,"Signal (S)","l");
	legend->AddEntry(signalFcnAB,"Signal (S)","l");
	legend->AddEntry(backFcnRD,"Background (B)","l");
	legend->AddEntry(backFcnAB,"Background (B)","l");

	legend->Draw();
	
	gStyle->SetOptStat(0);
	c->Update();
	
	
	static TF1 fitFcns[5]; // to be returned
	
	fitFcns[0] = *fitFcnRD;
	fitFcns[1] = *fitFcnAD;
	fitFcns[2] = *fitFcnRL005;
	fitFcns[3] = *fitFcnAL;
	fitFcns[4] = *fitFcnAB;
	
	return fitFcns;
		
}

