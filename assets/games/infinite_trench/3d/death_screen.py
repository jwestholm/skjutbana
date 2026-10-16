"""OpenGL 3.3 fullscreen death fade, downward blood streaks, and results card.

No offscreen framebuffer, external textures or extra packages. Text is rasterized
once with pygame.font, then drawn using the same ModernGL window/context.
"""
import math

BLOOD_VERT = '''#version 330
in vec2 in_pos;
out vec2 uv;
void main(){uv=in_pos*.5+.5; gl_Position=vec4(in_pos,0.0,1.0);}
'''
BLOOD_FRAG = '''#version 330
in vec2 uv;
uniform float death_age;
out vec4 color;
float hash(float x){return fract(sin(x*127.1+12.7)*43758.5453);}
void main(){
    float t=max(0.0,death_age);
    float shade=clamp(t/1.25,0.0,1.0)*.96;
    float edge=1.0-smoothstep(.30,.77,length((uv-.5)*vec2(1.2,1.0)));
    vec3 black=vec3(.025,.007,.009);
    float blood=0.0;
    // Red curtain creeps down from upper edge while uneven drips descend.
    float curtain=1.03-min(1.10,t*.57);
    for(int i=0;i<22;i++){
        float x=(float(i)+.35+.4*hash(float(i)+.3))/22.0;
        float width=.008+.017*hash(float(i)+14.0);
        float length=.11+.47*hash(float(i)+33.0);
        float head=curtain+length*clamp(t*.43,0.0,1.0);
        float stripe=(1.0-smoothstep(width*.65,width*1.5,abs(uv.x-x)));
        float tip=smoothstep(head-.12,head,uv.y);
        blood=max(blood,stripe*tip*.9*clamp(t*1.9,0.0,1.0));
    }
    blood=max(blood,smoothstep(curtain-.09,curtain+.03,uv.y)*clamp(t*1.8,0.0,1.0));
    // Entire frame goes dark; blood remains visible over the dark film.
    vec3 rgb=mix(black,vec3(.32,.009,.016),blood*.91);
    color=vec4(rgb,clamp(shade + blood*.70 + (1.0-edge)*.14*shade,0.0,.99));
}
'''
TEXT_VERT = '''#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 uv;
void main(){uv=in_uv;gl_Position=vec4(in_pos,0.0,1.0);}
'''
TEXT_FRAG = '''#version 330
uniform sampler2D tex;
uniform float opacity;
in vec2 uv;
out vec4 color;
void main(){vec4 sample_color=texture(tex,uv);color=vec4(sample_color.rgb,sample_color.a*opacity);}
'''


class DeathScreen:
    def __init__(self,ctx):
        import numpy as np
        self.ctx=ctx
        self.program=ctx.program(vertex_shader=BLOOD_VERT,fragment_shader=BLOOD_FRAG)
        self.text_program=ctx.program(vertex_shader=TEXT_VERT,fragment_shader=TEXT_FRAG)
        vertices=np.asarray([-1.,-1., 1.,-1., 1.,1., -1.,-1., 1.,1., -1.,1.],dtype='f4')
        self.vbo=ctx.buffer(vertices.tobytes())
        self.vao=ctx.vertex_array(self.program,[(self.vbo,'2f','in_pos')])
        quad=np.asarray([-1.,-1.,0.,0., 1.,-1.,1.,0., 1.,1.,1.,1.,
                         -1.,-1.,0.,0., 1.,1.,1.,1., -1.,1.,0.,1.],dtype='f4')
        self.text_vbo=ctx.buffer(quad.tobytes())
        self.text_vao=ctx.vertex_array(self.text_program,[(self.text_vbo,'2f 2f','in_pos','in_uv')])
        self.text_program['tex'].value=0
        self._texture=None
        self._key=None
        self._status_key=None
        self._status_texture=None

    def _make_card(self,stats,w,h):
        import pygame
        import moderngl
        card=pygame.Surface((w,h),pygame.SRCALPHA,32)
        large=pygame.font.Font(None,max(48,int(h*.115)))
        medium=pygame.font.Font(None,max(26,int(h*.061)))
        small=pygame.font.Font(None,max(22,int(h*.043)))
        lines=[('DU ÄR DÖD',large,(245,225,216)),
               (f'Du kom {stats.score_metres} meter',medium,(230,224,212)),
               (f'Dödade {stats.kills} fiender',medium,(230,224,212)),
               (f'Sköt {stats.shots} skott',medium,(230,224,212)),
               (f'TOTALA POÄNG: {stats.score}',medium,(245,194,155)),
               ('R: Starta om (samma seed)',small,(188,183,176)),
               ('S: Ny bana',small,(188,183,176))]
        line_heights=[int(font.get_height()*1.23) for _,font,_ in lines]
        start=max(14,(h-sum(line_heights))//2)
        for (text,font,color),advance in zip(lines,line_heights):
            glyph=font.render(text,True,color)
            shadow=font.render(text,True,(15,3,5))
            px=max(8,(w-glyph.get_width())//2)
            card.blit(shadow,(px+2,start+3))
            card.blit(glyph,(px,start))
            start+=advance
        if self._texture is not None:self._texture.release()
        self._texture=self.ctx.texture((w,h),4,pygame.image.tostring(card,'RGBA',True))
        self._texture.filter=(moderngl.LINEAR,moderngl.LINEAR)

    def draw(self,age,stats,w,h):
        import moderngl
        key=(w,h,stats.score_metres,stats.shots,stats.kills)
        if key!=self._key:
            self._make_card(stats,w,h)
            self._key=key
        self.ctx.disable(moderngl.DEPTH_TEST)
        try:
            self.program['death_age'].value=float(age)
            self.vao.render(moderngl.TRIANGLES)
            if age>.45:
                self.text_program['opacity'].value=min(1.,max(0.,(age-.45)/.85))
                self._texture.use(location=0)
                self.text_vao.render(moderngl.TRIANGLES)
        finally:
            self.ctx.enable(moderngl.DEPTH_TEST)

    def draw_status(self,lines,w,h):
        """Small cached status card using the existing text shader/quad."""
        import pygame
        import moderngl
        key=tuple(lines)
        width=min(640,max(200,w-24));height=26*len(lines)+10
        if self._status_key!=(key,width):
            surface=pygame.Surface((width,height),pygame.SRCALPHA,32)
            surface.fill((12,17,15,190))
            font=pygame.font.Font(None,23)
            for i,line in enumerate(lines):
                surface.blit(font.render(line,True,(235,237,222)),(8,5+i*26))
            if self._status_texture is not None:self._status_texture.release()
            self._status_texture=self.ctx.texture((width,height),4,pygame.image.tostring(surface,'RGBA',True))
            self._status_key=(key,width)
        viewport=self.ctx.viewport
        try:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.viewport=(12,max(0,h-height-12),width,height)
            self.text_program['opacity'].value=1.
            self._status_texture.use(location=0)
            self.text_vao.render(moderngl.TRIANGLES)
        finally:
            self.ctx.viewport=viewport
            self.ctx.enable(moderngl.DEPTH_TEST)

    def release(self):
        if self._status_texture is not None:self._status_texture.release()
        if self._texture is not None:self._texture.release()
        self.vao.release();self.vbo.release();self.program.release()
        self.text_vao.release();self.text_vbo.release();self.text_program.release()
