#include "msfunc.h"
#include <allegro.h>

//---------------------------------------------------
// This function draws a trace of color col when 
// we press the left button of the mouse, until 
// the ESC key is pressed
//---------------------------------------------------
void draw_mouse(int col)
{	
int x, y;
	show_mouse(screen);
	do {
		if (mouse_b & 1) {
			x = mouse_x;
			y = mouse_y;
			putpixel(screen, x, y, col);
		}
	} while (!key[KEY_ESC]);
}
