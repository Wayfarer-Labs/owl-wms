# import moderngl

# display = moderngl.create_context(standalone=True)
# gl_buffer = display.buffer(b"Hello World!")
# gl_buffer.read()
# b"Hello World!"
# if display.info["GL_VERSION"]:
#     gl_version = gl_buffer.version_code


# import moderngl
# import numpy as np
# from PIL import Image

# ctx = moderngl.create_standalone_context()
# prog = ctx.program(
#     vertex_shader="""
#         #version 410
#         in vec2 in_vert;
#         in vec3 in_color;
#         out vec3 v_color;

#         void main() {
#             v_color = in_color;
#             gl_Position = vec4(in_vert, 0.0, 1.0);
#         }
#     """,
#     fragment_shader="""
#         #version 410
#         in vec3 v_color;
#         out vec3 f_color;
#         void main() {
#             f_color = v_color;
#         }
#     """,
# )

# x = np.linspace(-1.0, 1.0, 50)
# y = np.random.rand(50) - 0.5
# r = np.ones(50)
# g = np.zeros(50)
# b = np.zeros(50)

# vertices = np.dstack([x, y, r, g, b])


# vbo = ctx.buffer(vertices.astype("f4").tobytes())
# vao = ctx.simple_vertex_array(prog, vbo, "in_vert", "in_color")

# fbo = ctx.simple_framebuffer((512, 512))
# fbo.use()
# fbo.clear(0.0, 0.0, 0.0, 1.0)
# vao.render(moderngl.LINE_STRIP)

# Image.frombytes("RGB", fbo.size, fbo.read(), "raw", "RGB", 0, -1).show()

# import moderngl

# ctx = moderngl.create_standalone_context()

# prog = ctx.program(
#     vertex_shader="""
#         #version 410
#         in vec2 in_vert;
#         in vec3 in_color;

#         out vec3 v_color;
#         void main() {
#             v_color = in_color;
#             gl_Position = vec4(in_vert, 0.0, 1.0);
#         }
#     """,
#     fragment_shader="""
#         #version 410

#         in vec3 v_color;
#         out vec3 f_color;

#         void main() {
#             f_color = v_color;
#         }
#     """,
# )

# import moderngl

# import numpy as np

# ctx = moderngl.create_standalone_context()
# prog = ctx.program(
#     vertex_shader="""
#         #version 410

#         in vec2 in_vert;
#         in vec3 in_color;
#         out vec3 v_color;

#         void main() {
#             v_color = in_color;
#             gl_Position = vec4(in_vert, 0.0, 1.0);
#         }
#     """,
#     fragment_shader="""
#         #version 410
#         in vec3 v_color;

#         out vec3 f_color;
#         void main() {
#             f_color = v_color;
#         }
#     """,
# )

# x = np.linspace(-1.0, 1.0, 50)
# y = np.random.rand(50) - 0.5
# r = np.ones(50)
# g = np.zeros(50)
# b = np.zeros(50)
# vertices = np.dstack([x, y, r, g, b])

# vbo = ctx.buffer(vertices.astype("f4").tobytes())
# vao = ctx.simple_vertex_array(prog, vbo, "in_vert", "in_color")


# import glfw
# from OpenGL.GL import *
# from OpenGL.GLU import *
# import numpy as np

# # ---------- Callback state ----------
# last_x, last_y =0,0
# rot_x, rot_y =0.0,0.0
# mouse_down = False

# def cursor_pos(window, xpos, ypos):
#     global last_x, last_y, rot_x, rot_y
#     if mouse_down:
#     dx = xpos - last_x
#     dy = ypos - last_y
#     rot_x += dy *0.5
#     rot_y += dx *0.5
#     last_x, last_y = xpos, ypos

# def mouse_button(window, button, action, mods):
#     global mouse_down
#     if button == glfw.MOUSE_BUTTON_LEFT:
#     mouse_down = (action == glfw.PRESS)

#     # ---------- Cube data ----------
#     vertices = np.array([
#     [-1, -1, -1], [1, -1, -1], [1,1, -1], [-1,1, -1],
#     [-1, -1,1], [1, -1,1], [1,1,1], [-1,1,1]
#     ], dtype='f')
#     faces = [
#     (0,1,2,3), (4,5,6,7), (0,1,5,4),
#     (2,3,7,6), (0,3,7,4), (1,2,6,5)
#     ]
#     colors = [
#     (1,0,0), (0,1,0), (0,0,1),
#     (1,1,0), (1,0,1), (0,1,1)
#     ]

#     # ---------- Main program ----------
#     if not glfw.init():
#     raise SystemExit
#     window = glfw.create_window(800,600, "3D Cube (glfw)", None, None)
#     glfw.make_context_current(window)
#     glfw.set_cursor_pos_callback(window, cursor_pos)
#     glfw.set_mouse_button_callback(window, mouse_button)

#     while not glfw.window_should_close(window):
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     glEnable(GL_DEPTH_TEST)
#     glLoadIdentity()
#     glTranslatef(0,0, -5)
#     glRotatef(rot_x,1,0,0)
#     glRotatef(rot_y,0,1,0)
#     glBegin(GL_QUADS)
#     for face, color in zip(faces, colors):
#     glColor3fv(color)
#     for idx in face:
#     glVertex3fv(vertices[idx])
#     glEnd()
#     glfw.swap_buffers(window)
#     glfw.poll_events()

#     glfw.terminate()
