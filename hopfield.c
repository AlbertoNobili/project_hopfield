#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <allegro.h>
#include <time.h>

#include "ptask.h"
#include "kbfunc.h"
#include "constant.h"

static int	hw[NMAX][NMAX];		// weight matrix
static int	ts[MMAX][NMAX];		// training set
static int	x[NMAX];			// state vector
static int	y[NMAX];			// state vector copy
static int	hn;					// # of neurons
static int	hm;					// # of memories
static int	rn;					// # of rows
static int	cn;					// # of columns
static int	mode;				// mode of evolution
static int	input;				// # of selected input (from 0 to 8)

void load_ts(char *name)
{
char	c;
int		i, k;
int		val;
FILE	*fp;

	fp = fopen(name, "r");
	if(fp == NULL){
		printf("FILE %s NOT FOUND\n", name);
		exit(1);
	}
	
	do {c = getc(fp);} while (c != ':');
	fscanf(fp, "%d", &hn);			// read # of neurons
	if(hn >NMAX){
		printf("Too many neurons\n");
		exit(1);
	}
	
	do {c = getc(fp);} while (c != ':');
	fscanf(fp, "%d %d", &rn, &cn);	// read # of rows and columns
	if(rn*cn != hn){
		printf("Rows-columns do not match\n");
		exit(1);
	}
	
	do {c = getc(fp);} while (c != ':');
	fscanf(fp, "%d", &hm);			// read # of examples
	if(hm > MMAX){
		printf("Too many memories\n");
		exit(1);
	}
	
	for(k=0; k<hm; k++){			// for each example k
		do {c = getc(fp);} while (c != ':');
		for(i=0; i<hn; i++){		// read all elements
			fscanf(fp, "%d", &val);
			ts[k][i] = 2*val -1;	// conv. 0,1 in -1,1
		}
	}
	
	fclose(fp);
} 

void init_weights()
{
int i, j, k;
	
	for (k=0; k<hm; k++){
		for(i=0; i<hn; i++)
			for(j=0; j<hn; j++)
				hw[i][j] += ts[k][i]*ts[k][j];
	}
	for (i=0; i<hn; i++) 
		hw[i][i] = 0;
}

float energy(int *x)
{
int		i, j;
float	E;

	E =0;
	for (i=0; i<hn-1; i++)
		for(j=i+1; j<hn; j++)
			E -= hw[i][j] * x[i] * x[j];
	E = E/(hm*hn*hn);	 // normalized in [0, 1]
	// hw[i][j] va da -hm a +hm; x[i] vale -1, +1
	// quindi E prima di essere normalizzato sta tra -hm*hn*hn/2 e +hm*hn*hn/2
	return E;
}

int hamming(int *x, int *y) // dice in quanti elementi differiscono i due stati
{
int	i, sum=0;

	for(i=0; i<hn; i++)
		if(x[i] != y[i])
			sum++;
	return sum;
}

void flip_bit(int k, int *x)
{
	x[k] = -x[k];
}

void state_copy(int *x, int *y)
{
int i;
	for (i=0; i<hn; i++) y[i] = x[i];
}

//---------------------------------------------
//	BINRAND: returns a random int in {0, 1}
//---------------------------------------------
int binrand ()
{
float	r;
	r = rand()/(float)RAND_MAX;	// rand in [0, 1)
	r = (r<0.5)? 0:1;
	return r;
}
//---------------------------------------------
//	INTRAND: returns a random int in [min, max)
//---------------------------------------------
int intrand (int min, int max)
{
float	r;
	r = rand()/(float)RAND_MAX;	// rand in [0, 1)
	r = min + r*(max-min);
	return (int)r;
}
//---------------------------------------------

void init_rand_status()
{
int i;
	
	for(i=0; i<hn; i++){
		x[i] = binrand();
	}
}

int evolve_sync(int *x, int *y) // evolve lo stato x e lo scrive in y e ritorna quanti elementi ha cambiato da x a y
{
int i, j;
int sum;
	
	for (i=0; i<hn; i++){
		sum = 0;				// compute activation
		for (j=0; j<hn; j++)
			sum += hw[i][j];
		if(sum < 0)	y[i] = -1;	// compute output
		else		y[i] = 1;
	}
	return hamming(x, y);
}

int evolve_async(int *x, int *y)
{
int		i;
int		z[NMAX];	// state after a sync transition
int 	hdist;		// hamming distance os sync transition
int		win;		// element leading to min energy
float	e, min;		// energy and min energy

	state_copy(x, y);			// prepare new state
	hdist = evolve_sync(x, z);	// make sync transition
	if(hdist == 0) return 0;	// stable state
	
	// Otherwise find among the possible transitions
	// the one with the minimum energy
	
	min = INT_MAX;
	for (i=0; i<hn; i++){		// for each neuron
		if (z[i] != x[i]) {		// if the neuron can change
			flip_bit(i, y);		// try async transition
			e = energy(y);
			if(e < min){
				min = e;
				win = i;
			}
			flip_bit(i, y);
		}
	}
	flip_bit(win, y);
	return 1;
}

void print_state_grid(int xbase, int ybase, int scale);

int evolve_net(int *x, int *y, int mode) // evolve la rete dallo stato x e salva lo stato finale sia in x che in y
{
int	h = 0;
int count = 0;	// transitin counter

	do {
		print_state_grid(XMED + 20, BRD + 50, 16); //comment this line to speed up 
		//print_energy(x);
		if (mode == SYNC)	h = evolve_sync(x, y);
		else				h = evolve_async(x, y);
		if (h == 0)	printf("Stable state\n");
		count++;
		state_copy(y, x);
	} while (h > 0);
	return count;
}

void add_noise()
{
int i, k;
	
	for(i=0; i<NFLIP; i++){
		k = intrand(0, hn);
		flip_bit(k, x);
	}
	print_state_grid(XMED + 20, BRD + 50, 16);
}

void print_state_ungrid(int xbase, int ybase, int scale) 
{
int	i, j;
	//printf("la prima ascissa vale %d\n", xbase);
	//printf("row number: %d\ncolumn number: %d\n", rn, cn);

	for(i=0; i<rn; i++){
		for(j=0; j<cn; j++){
			if(x[i*cn+j]==1)
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale, ybase+i*scale+scale, 12);
			else
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale, ybase+i*scale+scale, 15);
		}
	}
	//printf("l'ultima ascissa vale %d\n", xbase+j*16+15);
}

void print_state_ungrid_dark(int xbase, int ybase, int scale) 
{
int	i, j;
	//printf("la prima ascissa vale %d\n", xbase);
	//printf("row number: %d\ncolumn number: %d\n", rn, cn);

	for(i=0; i<rn; i++){
		for(j=0; j<cn; j++){
			if(x[i*cn+j]==1)
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale, ybase+i*scale+scale, 12);
			else
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale, ybase+i*scale+scale, 8);
		}
	}
	//printf("l'ultima ascissa vale %d\n", xbase+j*16+15);
}

void print_state_grid(int xbase, int ybase, int scale) 
{
int	i, j;
	//printf("la prima ascissa vale %d\n", xbase);
	//printf("row number: %d\ncolumn number: %d\n", rn, cn);

	for(i=0; i<rn; i++){
		for(j=0; j<cn; j++){
			if(x[i*cn+j]==1)
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale-1, ybase+i*scale+scale-1, 12);
			else
				rectfill(screen, xbase+j*scale+1, ybase+i*scale+1, xbase+j*scale+scale-1, ybase+i*scale+scale-1, 15);
		}
	}
	//printf("l'ultima ascissa vale %d\n", xbase+j*16+15);
}

void display_comands()
{
char	s[30];

	sprintf(s, "ARROWS  select input");
	textout_ex(screen, font, s, XMED+10, YMED+40, MCOL, BKG);
	sprintf(s, "N       add noise");
	textout_ex(screen, font, s, XMED+10, YMED+55, MCOL, BKG);
	sprintf(s, "E       evolve net");
	textout_ex(screen, font, s, XMED+10, YMED+70, MCOL, BKG);
	sprintf(s, "A       add memory");
	textout_ex(screen, font, s, XMED+10, YMED+85, MCOL, BKG);
	sprintf(s, "D       delete memory");
	textout_ex(screen, font, s, XMED+10, YMED+100, MCOL, BKG);
	sprintf(s, "M       change mode");
	textout_ex(screen, font, s, XMED+10, YMED+115, MCOL, BKG);
	sprintf(s, "ESC     exit");
	textout_ex(screen, font, s, XMED+10, YMED+130, MCOL, BKG);
	
}

void display_ts()
{
int i,j;
int xbase, ybase;

	xbase = BRD + 5;
	ybase = BRD + 50;
	load_ts("lettere.dat");
	for(i=0; i<3; i++){
		for(j=0; j<9; j++){
			state_copy(x, y);
			state_copy(&ts[i*9+j][0], x);
			print_state_ungrid(xbase + 5 + 53*j, ybase+5+50*i, 3);
			if (j+i*9 == input){
				print_state_ungrid_dark(xbase + 5 + 53*j, ybase+5+50*i, 3);
			}
			state_copy(y, x);
		}
	}
}

void aggiorna_status()
{	
	state_copy(&ts[input][0], x);
	print_state_grid(XMED + 20, BRD + 50, 16);
}

void display()
{
char	s[20];

	allegro_init();
	set_color_depth(8);
	set_gfx_mode(GFX_AUTODETECT_WINDOWED, XWIN, YWIN, 0, 0);
	clear_to_color(screen, BKG);
	install_keyboard();
	
	srand(time(NULL));
	
	rect(screen, BRD, BRD, XMED, YMED, MCOL);
	sprintf(s, "TRAINING SET");
	textout_centre_ex(screen, font, s, XC1, YC1, MCOL, BKG);
	
	rect(screen, BRD, YMED, XMED, YBOX, MCOL);
	sprintf(s, "CURRENT STATE");
	textout_centre_ex(screen, font, s, XC2, YC1, MCOL, BKG);
	
	rect(screen, XMED, BRD, XBOX, YMED, MCOL);
	sprintf(s, "NETWORK MONITOR");
	textout_centre_ex(screen, font, s, XC1, YC2, MCOL, BKG);
	
	rect(screen, XMED, YMED, XBOX, YBOX, MCOL);
	sprintf(s, "USER COMANDS");
	textout_centre_ex(screen, font, s, XC2, YC2, MCOL, BKG);
	
	display_comands();
	display_ts();
	aggiorna_status();

	return;
}

void command_interpreter()
{
char	scan;
	
	do{
		scan = get_scancode_nb();
		switch (scan){
			case KEY_E:		evolve_net(x, y, mode);
							break;
			case KEY_N:		add_noise();
							break;
			case KEY_M:		mode = (mode+1)%2;
							break;
			case KEY_RIGHT:	input = (input+1)%27;
							aggiorna_status();
							display_ts();
							break;
			case KEY_LEFT:	input = (input+26)%27;
							aggiorna_status();
							display_ts();
							break;
			case KEY_UP:	input = (input+18)%27;
							aggiorna_status();
							display_ts();
							break;
			case KEY_DOWN:	input = (input+9)%27;
							aggiorna_status();
							display_ts();
							break;
							
			default:	break;
		}
	} while (scan != KEY_ESC);
}

int main (int argc, char *argv[])
{
char	tsname[15];				// training set name
	
	strcpy(tsname, "test.dat");	// default name
	if (argc > 1)	strcpy(tsname, argv[1]);	// specified name
	
	mode = ASYNC;
	input = 0;
	load_ts(tsname);
	init_weights();
	display();
	command_interpreter();
	allegro_exit();
	return 0;
}











