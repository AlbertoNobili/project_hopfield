#include "kbfunc.h"
#include <allegro.h>
#include <stdio.h>


//------------------------------------------------------
//	The following function waits for a key pressed and 
//	extracts the corresponding ascii code and scan code
//------------------------------------------------------
void get_keycodes (char *scan, char *ascii)
{
int	k;

	k =readkey();
	*ascii = k;
	*scan = k >> 8;
}

//------------------------------------------------------
//	The following function returns the scancode of a 
//	pressed key (non blocking)
//------------------------------------------------------
char get_scancode_nb ()
{
	if (keypressed())
		return readkey() >> 8;
	else 
		return 0;
}

//------------------------------------------------------
//	The following function reads a string from the keyboard
//	and dislplays the echo in graphic mode at position (x,y)
//	with color c and background b
//------------------------------------------------------
void get_string(char *str, int x, int y, int c, int b)
{
char	ascii , scan, s[2];
int		i = 0;

	do {
		get_keycodes(&scan, &ascii);
		if (scan != KEY_ENTER) {
			s[0] = ascii;		// put ascii in s for echoing
			s[1] = '\0';
			textout_ex(screen, font, s, x, y, c, b);	// echo
			x = x + 8;
			str[i++] = ascii;	//insert character in string
		}
	} while (scan != KEY_ENTER);
	str[i] = '\0';
}

//-----------------------------------------------------
// This function reads a float from the keyboard and 
// stores it in the variable x
//-----------------------------------------------------
void read_float(float *x)
{
char	str[20];
	
	textout_ex(screen, font, "inserire un numero: x = ", 10 , 30 , 3, 0);
	get_string(str, 202, 30, 3, 0);
	sscanf(str, "%f", x);
}

//gcc -c kbfunc.c
	
		
		
		
		
		
		
		
		
		
