#include <windows.h>
#include <ddraw.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct
{
	uint32_t Width, Height, Depth;
	uint8_t *Data;
} VkuImage_t;

LPDIRECTDRAW lpDD=NULL;
LPDIRECTDRAWSURFACE lpDDSFront=NULL;
LPDIRECTDRAWSURFACE lpDDSBack=NULL;
HWND hWnd=NULL;

char *szAppName="DirectDraw";

RECT WindowRect;

int Width=800, Height=600;

bool Done=false, Key[256];

extern float *output;

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void Render(void);
int CreateDDraw(void);
void DestroyDDraw(void);

void *CreateDDrawWindow(void *Arg)
{
	HINSTANCE hInstance=GetModuleHandle(NULL);
	WNDCLASS wc;
	wc.style=CS_VREDRAW|CS_HREDRAW|CS_OWNDC;
	wc.lpfnWndProc=WndProc;
	wc.cbClsExtra=0;
	wc.cbWndExtra=0;
	wc.hInstance=hInstance;
	wc.hIcon=LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor=LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground=GetStockObject(BLACK_BRUSH);
	wc.lpszMenuName=NULL;
	wc.lpszClassName=szAppName;

	RegisterClass(&wc);

	RECT WindowRect;
	WindowRect.left=0;
	WindowRect.right=Width*4;
	WindowRect.top=0;
	WindowRect.bottom=Height*4;

	AdjustWindowRect(&WindowRect, WS_POPUP, FALSE);

	hWnd=CreateWindow(szAppName, szAppName, WS_POPUP, CW_USEDEFAULT, CW_USEDEFAULT, WindowRect.right-WindowRect.left, WindowRect.bottom-WindowRect.top, NULL, NULL, hInstance, NULL);

	ShowWindow(hWnd, SW_SHOW);
	SetForegroundWindow(hWnd);

	if(!CreateDDraw())
	{
		DestroyWindow(hWnd);
		return NULL;
	}

	MSG msg={ 0 };
	while(!Done)
	{
		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if(msg.message==WM_QUIT)
				Done=1;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			RECT RectSrc, RectDst;
			POINT Point={ 0, 0 };

			if(IDirectDrawSurface7_IsLost(lpDDSBack)==DDERR_SURFACELOST)
				IDirectDrawSurface7_Restore(lpDDSBack);

			if(IDirectDrawSurface7_IsLost(lpDDSFront)==DDERR_SURFACELOST)
				IDirectDrawSurface7_Restore(lpDDSFront);

			Render();

			ClientToScreen(hWnd, &Point);
			GetClientRect(hWnd, &RectDst);
			OffsetRect(&RectDst, Point.x, Point.y);
			SetRect(&RectSrc, 0, 0, Width, Height);

			IDirectDrawSurface7_Blt(lpDDSFront, &RectDst, lpDDSBack, &RectSrc, DDBLT_WAIT, NULL);
		}
	}

	DestroyDDraw();
	DestroyWindow(hWnd);

	return NULL;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch(uMsg)
	{
		case WM_CREATE:
			break;

		case WM_CLOSE:
			PostQuitMessage(0);
			break;

		case WM_DESTROY:
			break;

		case WM_SIZE:
			break;

		case WM_KEYDOWN:
			Key[wParam]=TRUE;
			break;

		case WM_KEYUP:
			Key[wParam]=FALSE;
			break;
	}

	return DefWindowProc(hWnd, uMsg, wParam, lParam);
}

void ClearSurface(LPDIRECTDRAWSURFACE lpDDS)
{
	DDBLTFX ddbltfx;

	ddbltfx.dwSize=sizeof(DDBLTFX);
	ddbltfx.dwFillColor=0x00000000;

	if(IDirectDrawSurface7_Blt(lpDDS, NULL, NULL, NULL, DDBLT_COLORFILL, &ddbltfx)!=DD_OK)
		return;
}

void point(DDSURFACEDESC2 ddsd, uint32_t x, uint32_t y, float c[3])
{
	if(x<1)
		return;
	if(x>ddsd.dwWidth-1)
		return;
	if(y<1)
		return;
	if(y>ddsd.dwHeight-1)
		return;

	int i=(y*ddsd.lPitch+x*(ddsd.ddpfPixelFormat.dwRGBBitCount>>3));

	((uint8_t *)ddsd.lpSurface)[i+0]=(unsigned char)(c[2]*255.0f)&0xFF;
	((uint8_t *)ddsd.lpSurface)[i+1]=(unsigned char)(c[1]*255.0f)&0xFF;
	((uint8_t *)ddsd.lpSurface)[i+2]=(unsigned char)(c[0]*255.0f)&0xFF;
}

#define INPUT_SIZE 2
#define HIDDEN_SIZE 14
#define OUTPUT_SIZE 3

extern float *copy_input_hidden_weights, *copy_input_hidden_biases;
extern float *copy_hidden_hidden_weights, *copy_hidden_hidden_biases;
extern float *copy_hidden_output_weights, *copy_hidden_output_biases;

float hidden_output[HIDDEN_SIZE], hidden2_output[HIDDEN_SIZE];

void Render(void)
{
	DDSURFACEDESC2 ddsd;
	HRESULT ret=DDERR_WASSTILLDRAWING;

	ClearSurface(lpDDSBack);

	memset(&ddsd, 0, sizeof(DDSURFACEDESC2));
	ddsd.dwSize=sizeof(DDSURFACEDESC2);

	while(ret==DDERR_WASSTILLDRAWING)
		ret=IDirectDrawSurface7_Lock(lpDDSBack, NULL, &ddsd, 0, NULL);

	for(uint32_t y=0;y<ddsd.dwHeight;y++)	
	{
		float dy=(float)y/(float)ddsd.dwHeight;

		for(uint32_t x=0;x<ddsd.dwWidth;x++)
		{
			float dx=(float)x/(float)ddsd.dwWidth;

			float input[INPUT_SIZE]={ dx, dy };
			float output[OUTPUT_SIZE];

			forward_propagation(input,
								copy_input_hidden_weights, copy_input_hidden_biases, hidden_output,
								copy_hidden_hidden_weights, copy_hidden_hidden_biases, hidden2_output,
								copy_hidden_output_weights, copy_hidden_output_biases, output);

			point(ddsd, x, y, (float[3]) { output[2], output[1], output[0] });
		}
	}

	IDirectDrawSurface7_Unlock(lpDDSBack, NULL);
}

BOOL CreateDDraw(void)
{
	DDSURFACEDESC2 ddsd;
	LPDIRECTDRAWCLIPPER lpClipper=NULL;

	if(DirectDrawCreateEx(NULL, &lpDD, &IID_IDirectDraw7, NULL)!=DD_OK)
	{
		MessageBox(hWnd, "DirectDrawCreateEx failed.", "Error", MB_OK);
		return FALSE;
	}

	if(IDirectDraw7_SetCooperativeLevel(lpDD, hWnd, DDSCL_NORMAL)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDraw7_SetCooperativeLevel failed.", "Error", MB_OK);
		return FALSE;
	}

	memset(&ddsd, 0, sizeof(ddsd));
	ddsd.dwSize=sizeof(ddsd);
	ddsd.dwFlags=DDSD_CAPS;
	ddsd.ddsCaps.dwCaps=DDSCAPS_PRIMARYSURFACE;

	if(IDirectDraw7_CreateSurface(lpDD, &ddsd, &lpDDSFront, NULL)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDraw7_CreateSurface (Front) failed.", "Error", MB_OK);
		return FALSE;
	}

	ddsd.dwFlags=DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS;
	ddsd.dwWidth=Width;
	ddsd.dwHeight=Height;
	ddsd.ddsCaps.dwCaps=DDSCAPS_OFFSCREENPLAIN;

	if(IDirectDraw7_CreateSurface(lpDD, &ddsd, &lpDDSBack, NULL)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDraw7_CreateSurface (Back) failed.", "Error", MB_OK);
		return FALSE;
	}

	if(IDirectDraw7_CreateClipper(lpDD, 0, &lpClipper, NULL)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDraw7_CreateClipper failed.", "Error", MB_OK);
		return FALSE;
	}

	if(IDirectDrawClipper_SetHWnd(lpClipper, 0, hWnd)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDrawClipper_SetHWnd failed.", "Error", MB_OK);
		return FALSE;
	}

	if(IDirectDrawSurface_SetClipper(lpDDSFront, lpClipper)!=DD_OK)
	{
		MessageBox(hWnd, "IDirectDrawSurface_SetClipper failed.", "Error", MB_OK);
		return FALSE;
	}

	if(lpClipper!=NULL)
	{
		IDirectDrawClipper_Release(lpClipper);
		lpClipper=NULL;
	}

	return TRUE;
}

void DestroyDDraw(void)
{
	if(lpDDSBack!=NULL)
	{
		IDirectDrawSurface7_Release(lpDDSBack);
		lpDDSBack=NULL;
	}

	if(lpDDSFront!=NULL)
	{
		IDirectDrawSurface7_Release(lpDDSFront);
		lpDDSFront=NULL;
	}

	if(lpDD!=NULL)
	{
		IDirectDraw7_Release(lpDD);
		lpDD=NULL;
	}
}
