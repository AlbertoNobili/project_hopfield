#ifndef	KEYBOARD_H
#define	KEYBOARD_H
void get_keycodes (char *scan, char *ascii);
char get_scancode_nb();
void get_string(char *str, int x, int y, int c, int b);
void read_float(float *x);
#endif
