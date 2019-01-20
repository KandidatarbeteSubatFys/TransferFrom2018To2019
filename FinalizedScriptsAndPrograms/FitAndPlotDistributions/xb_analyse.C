
/* To run standalone:
 g++ -o scripts/xb_analyse scripts/xb_analyse.C `root-config --cflags --glibs`
 scripts/xb_analyse
*/


#define xb_analyse_cxx
#include "xb_analyse.h"
#include <TH1.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TApplication.h>
#include <TRint.h>

struct xb_particle
{
  double theta;
  double e;
  int   main_c;
  int   num_c;
};

struct xb_particles
{
  int n;
  xb_particle particles[162]; // too many
};

xb_particles _xb_particles;

void read_txt_event(FILE *fid, xb_particles *particles, int expectevent)
{
  char line[1024];

  fgets(line, sizeof (line), fid);

  int eventno;
  int n;

  n = sscanf(line, "Event: %d\n", &eventno);

  if (n != 1)
    {
      fprintf (stderr, "Failed to parse event header (#%d).  Got: '%s'\n",
	       expectevent, line);
      exit(1);
    }

  if (eventno != expectevent)
    {
      fprintf (stderr, "Mismatched event no, got #%d, expected (#%d)\n",
	       eventno, expectevent);
      exit(1);
    }

  particles->n = 0;

  for ( ; ; )
    {
      fgets(line, sizeof (line), fid);

      if (strcmp(line, "\n") == 0)
	break;

      xb_particle *part = &particles->particles[particles->n];

      n = sscanf(line,"  gamma: e=%lf  theta=%lf  main_c=%d num_c=%d\n",
		 &part->e,
		 &part->theta,
		 &part->main_c,
		 &part->num_c);

      if (n != 4)
	{
	  fprintf (stderr, "Failed to parse particle (event, #%d).  "
		   "Got: '%s'\n",
		   expectevent, line);
	  exit(1);
	}

      particles->n++;
    }
}

void xb_analyse::Loop(const char *txtfilename, TH1F *hist)
{
//   In a ROOT session, you can do:
//      root> .L xb_analyse.C
//      root> xb_analyse t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch

  if (fChain == 0) return;

  FILE *fid = NULL;

  if (txtfilename)
    fid = fopen(txtfilename, "r");

  int treated = 0;
  int particles = 0;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;

    if (txtfilename)
      {
	read_txt_event(fid, &_xb_particles, eventno);
    
	/*printf ("\n"); */
	for (int i = 0; i < _xb_particles.n; i++)
	  {
	    /*	
		printf ("%.2f  %d\n",
		_xb_particles.particles[i].e,
		_xb_particles.particles[i].main_c);
		printf ("entries: %.1f\n", hist->GetEntries());
	    */

	    xb_particle *part = &_xb_particles.particles[i];

	    double beta = dummyboostbetaz;

	    double doppler =
	      sqrt(1 - beta * beta) /
	      (1 - beta * cos(part->theta * M_PI / 180.0));

	    /* printf ("%.2f %.2f\n", part->theta, doppler); */
	
	    hist->Fill(part->e / doppler);
	  }
      }
    else
      {
	/* Data directly from root file. */

	for (int i = 0; i < gunn; i++)
	  {
	    /* Theta angle is that of the particle. */

	    double pxyz = sqrt(gunpx[i] * gunpx[i] +
			       gunpy[i] * gunpy[i] +
			       gunpz[i] * gunpz[i]);

	    double cos_theta = gunpz[i] / pxyz;

	    double theta = acos(cos_theta);

	    // 0.28 ?
	    cos_theta = cos(theta + 0.20*(-0.5+rand()*(1.0/RAND_MAX)));

	    double beta = dummyboostbetaz;

	    double doppler = sqrt(1 - beta * beta) /
	      (1 - beta * cos_theta);

	    // hist->Fill(gunT[i] / doppler);
	    hist->Fill(gunedepXB[i] / doppler);
	  }
      }

    treated++;
    particles += _xb_particles.n;
  }

  if (txtfilename)
    fclose(fid);

  printf ("# Treated: %d  particles: %d\n",
	  treated, particles);
}

void analyse_file(TCanvas *c1, xb_analyse &xba,
		  const char *txtfilename, TH1F *hist)
{
  xba.Loop(txtfilename, hist);

  printf ("entries: %.1f\n", hist->GetEntries());

  hist->Draw("same");
}

void do_analyse(const char *simcase, TCanvas *c1)
{
  printf (" *** Analyse: %s\n", simcase);
  
  for (int addbacki = -1; addbacki < 12; addbacki++)
    {
      char filename[256];
      char histname[64];
      char histtitle[64];
      const char *addback = NULL;
      int color = 1;

      switch (addbacki)
	{
	case -1: addback = "det"; color = 1; break;
	case 0: addback = "1-neigh-highest"; color = 2; break;	  
	case 1: addback = "2-neigh-highest"; color = 5; continue;
	case 2: addback = "2-chain-highest"; color = 5; continue;
	case 3: addback = "1-neigh-weighted-theta"; color = 3; continue;	  
	case 4: addback = "2-neigh-weighted-theta"; color = 5; continue;
	case 5: addback = "2-chain-weighted-theta"; color = 5; continue;
	case 6: addback = "rel_loss"; color = 6; break;
	case 7: addback = "abs_rel_loss"; color = 46; break;
	case 8: addback = "abs_loss"; color = 7; break;
	case 9: addback = "relative_lambda0_05"; color = 8; break;
	case 10: addback = "relative_lambda1"; color = 9; break;
	case 11: addback = "conv_compare"; color = 49; continue;
	}

      sprintf (filename, "xb_sim/addback/%s/xb_line_%s.txt",
	       addback, simcase);

      char simcasemangle[64];
      strcpy(simcasemangle, simcase);
      char *p;
      while ((p = strchr(simcasemangle, '.')) != NULL)
	*p = '_';

      sprintf (histname, "h_%s_%s", addback, simcasemangle);
      sprintf (histtitle, "h_%s_%s_title", addback, simcasemangle);

      TH1F *h = new TH1F(histname, histtitle, 700, 0., 7.);
      h->SetLineColor(color);

      char rootfile[256];

      sprintf (rootfile, "xb_sim/sim/xb_line_%s.root", simcase);

      xb_analyse xba(rootfile);

      analyse_file(c1, xba,
		   addbacki == -1 ? NULL : filename, h);
    }
}

#define countof(x) (sizeof(x)/sizeof(x[0]))

#ifndef __CINT__
int main(int argc, char **argv)
{
  //TApplication theApp("tapp", &argc, argv);
  TRint theApp("tapp", &argc, argv);

  TCanvas *c1 = new TCanvas("c1");

  const char *cases[] = { "E0.1MeV",
			  "E0.2MeV",
			  "E0.5MeV",
		      "E1.0MeV",
			  "E1.5MeV",
			  "E2.0MeV",
			  "E2.5MeV",
			  "E3.5MeV",
			  "E5.0MeV" };
	//const char *cases[] = {"E2.5MeV"};

  c1->Divide(3,3);

  for (int i = 0; i < countof(cases); i++)
    {
      c1->cd(i+1);
      do_analyse(cases[i], c1);
      gPad->BuildLegend();
      c1->Update();
    }
    

  // sleep(1000);

  theApp.Run();

  return 0; 
}
#endif
