import glfw, vulkan as vk

# ---- GLFW window ----
if not glfw.init():
    raise SystemExit
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
win = glfw.create_window(800, 600, "Vulkan on macOS", None, None)

# ---- Vulkan instance ----
extensions = glfw.get_required_instance_extensions() + [vk.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME]
app_info = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO, pApplicationName="Demo", applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0), pEngineName="NoEngine", engineVersion=vk.VK_MAKE_VERSION(1, 0, 0), apiVersion=vk.VK_API_VERSION_1_0)
create_info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app_info, enabledExtensionCount=len(extensions), ppEnabledExtensionNames=extensions, flags=vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)
instance = vk.vkCreateInstance(create_info, None)

# ---- Surface ----
surface = glfw.create_window_surface(instance, win, None, vk.VK_EXT_metal_surface)

# ---- Physical device (pick first supporting portability) ----
phys_devs = vk.vkEnumeratePhysicalDevices(instance)
pd = next(d for d in phys_devs if vk.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME in vk.vkEnumerateDeviceExtensionProperties(d, None))

# ---- Logical device ----
queue_family = 0
device_ext = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME, vk.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME]
queue_info = vk.VkDeviceQueueCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=queue_family, queueCount=1, pQueuePriorities=[1.0])
dev_info = vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, queueCreateInfoCount=1, pQueueCreateInfos=queue_info, enabledExtensionCount=len(device_ext), ppEnabledExtensionNames=device_ext)
device = vk.vkCreateDevice(pd, dev_info, None)

# ---- Simple render loop (clear to blue) ----
while not glfw.window_should_close(win):
    glfw.poll_events()
    # Acquire image, begin command buffer, clear, end, present (omitted for brevity)
    pass

vk.vkDestroyDevice(device, None)
vk.vkDestroySurfaceKHR(instance, surface, None)
vk.vkDestroyInstance(instance, None)
glfw.terminate()

# import glfw, vulkan as vk, sys

# # ---------- GLFW window ----------
# if not glfw.init():
#     sys.exit()
# glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
# win = glfw.create_window(800, 600, "Vulkan on macOS", None, None)

# # ---------- Vulkan instance ----------
# app_info = vk.VkApplicationInfo(pApplicationName="Demo", applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0), pEngineName="NoEngine", engineVersion=vk.VK_MAKE_VERSION(1, 0, 0), apiVersion=vk.VK_API_VERSION_1_0)
# inst = vk.vkCreateInstance(vk.InstanceCreateInfo(pApplicationInfo=app_info), None)

# # ---------- Surface (MoltenVK) ----------
# surface = vk.create_macos_surface_mvk(inst, vk.MacosSurfaceCreateInfoMVK(pView=glfw.get_cocoa_window(win)))

# # ---------- Physical device & queue ----------
# phys = next(dev for dev in vk.enumerate_physical_devices(inst) if vk.get_physical_device_surface_support_khr(dev, 0, surface))
# queue_family = 0
# device = vk.create_device(phys, vk.DeviceCreateInfo(queueCreateInfoCount=1, pQueueCreateInfos=[vk.DeviceQueueCreateInfo(queueFamilyIndex=queue_family, queueCount=1, pQueuePriorities=[1.0])]), None)
# queue = vk.get_device_queue(device, queue_family, 0)

# # ---------- Swapchain ----------
# swapchain = vk.create_swapchain_khr(
#     device,
#     vk.SwapchainCreateInfoKHR(
#         surface=surface,
#         minImageCount=2,
#         imageFormat=vk.FORMAT_B8G8R8A8_SRGB,
#         imageColorSpace=vk.COLOR_SPACE_SRGB_NONLINEAR_KHR,
#         imageExtent=vk.Extent2D(800, 600),
#         imageArrayLayers=1,
#         imageUsage=vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
#         preTransform=vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
#         presentMode=vk.PRESENT_MODE_FIFO_KHR,
#         clipped=True,
#     ),
#     None,
# )
# imgs = vk.get_swapchain_images_khr(device, swapchain)

# # ---------- Render pass & framebuffers ----------
# attachment = vk.AttachmentDescription(
#     format=vk.FORMAT_B8G8R8A8_SRGB, samples=vk.SAMPLE_COUNT_1_BIT, loadOp=vk.ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.ATTACHMENT_STORE_OP_STORE, initialLayout=vk.IMAGE_LAYOUT_UNDEFINED, finalLayout=vk.IMAGE_LAYOUT_PRESENT_SRC_KHR
# )
# subpass = vk.SubpassDescription(colorAttachmentCount=1, pColorAttachments=[vk.AttachmentReference(attachment=0, layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)])
# render_pass = vk.create_render_pass(device, vk.RenderPassCreateInfo(attachmentCount=1, pAttachments=[attachment], subpassCount=1, pSubpasses=[subpass]), None)
# framebuffers = [vk.create_framebuffer(device, vk.FramebufferCreateInfo(renderPass=render_pass, attachmentCount=1, pAttachments=[img], width=800, height=600, layers=1), None) for img in imgs]

# # ---------- Command pool & buffer ----------
# cmd_pool = vk.create_command_pool(device, vk.CommandPoolCreateInfo(queueFamilyIndex=queue_family), None)
# cmd_buf = vk.allocate_command_buffers(device, vk.CommandBufferAllocateInfo(commandPool=cmd_pool, level=vk.COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]

# # ---------- Main loop ----------
# while not glfw.window_should_close(win):
#     glfw.poll_events()
#     # record commands
#     vk.begin_command_buffer(cmd_buf, vk.CommandBufferBeginInfo())
#     vk.cmd_begin_render_pass(
#         cmd_buf,
#         vk.RenderPassBeginInfo(
#             renderPass=render_pass, framebuffer=framebuffers[0], renderArea=vk.Rect2D(offset=vk.Offset2D(0, 0), extent=vk.Extent2D(800, 600)), clearValueCount=1, pClearValues=[vk.ClearValue(color=vk.ClearColorValue(float32=[0.1, 0.2, 0.3, 1.0]))]
#         ),
#         vk.SUBPASS_CONTENTS_INLINE,
#     )
#     vk.cmd_end_render_pass(cmd_buf)
#     vk.end_command_buffer(cmd_buf)
#     # submit & present
#     vk.queue_submit(queue, 1, [vk.SubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd_buf])], vk.NULL_HANDLE)
#     vk.queue_present_khr(queue, vk.PresentInfoKHR(swapchainCount=1, pSwapchains=[swapchain], pImageIndices=[0]))
#     vk.queue_wait_idle(queue)

# # ---------- Cleanup ----------
# vk.destroy_device(device, None)
# vk.destroy_instance(inst, None)
# glfw.destroy_window(win)
# glfw.terminate()

# # import glfw
# # import vulkan as vk
# # import sys

# # # ---------- GLFW window ----------
# # if not glfw.init():
# #     sys.exit()
# # # Tell GLFW to not create an OpenGL context
# # glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
# # window = glfw.create_window(800, 600, "MoltenVK Demo", None, None)

# # # ---------- Vulkan instance ----------
# # app_info = vk.VkApplicationInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
# #     pApplicationName="MoltenVK Demo",
# #     applicationVersion=vk.VK_MAKE_VERSION(1, 4, 38),
# #     pEngineName="No Engine",
# #     engineVersion=vk.VK_MAKE_VERSION(1, 4, 38),
# #     # apiVersion=vk,
# # )
# # # Enable MoltenVK surface extension
# # extensions = [vk.VK_KHR_SURFACE_EXTENSION_NAME, vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME]
# # instance_info = vk.VkInstanceCreateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
# #     pApplicationInfo=app_info,
# #     enabledExtensionCount=len(extensions),
# #     ppEnabledExtensionNames=extensions,
# # )
# # instance = vk.vkCreateInstance(instance_info, None)

# # # ---------- Create macOS surface ----------
# # # Obtain the native NSWindow handle from GLFW
# # ns_window = glfw.get_cocoa_window(window)
# # surface_create_info = vk.VkMacOSSurfaceCreateInfoMVK(
# #     sType=vk.VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
# #     pView=ns_window,
# # )
# # surface = vk.vkCreateMacOSSurfaceMVK(instance, surface_create_info, None)

# # # ---------- Physical device & queue ----------
# # phys_devs = vk.vkEnumeratePhysicalDevices(instance)
# # phys_dev = phys_devs[0]
# # # Find a graphics+present queue family
# # queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev)
# # graphics_family = None
# # for i, q in enumerate(queue_families):
# #     present = vk.vkGetPhysicalDeviceSurfaceSupportKHR(phys_dev, i, surface)
# #     if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT and present:
# #         graphics_family = i
# #     break
# # assert graphics_family is not None

# # # ---------- Logical device ----------
# # queue_priority = 1.0
# # device_queue_info = vk.VkDeviceQueueCreateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
# #     queueFamilyIndex=graphics_family,
# #     queueCount=1,
# #     pQueuePriorities=[queue_priority],
# # )
# # device_extensions = [vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME]
# # device_info = vk.VkDeviceCreateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
# #     queueCreateInfoCount=1,
# #     pQueueCreateInfos=device_queue_info,
# #     enabledExtensionCount=len(device_extensions),
# #     ppEnabledExtensionNames=device_extensions,
# # )
# # device = vk.vkCreateDevice(phys_dev, device_info, None)
# # graphics_queue = vk.vkGetDeviceQueue(device, graphics_family, 0)

# # # ---------- Swapchain ----------
# # surface_caps = vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev, surface)
# # format_props = vk.vkGetPhysicalDeviceSurfaceFormatsKHR(phys_dev, surface)
# # surface_format = format_props[0]
# # present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
# # swapchain_info = vk.VkSwapchainCreateInfoKHR(
# #     sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
# #     surface=surface,
# #     minImageCount=surface_caps.minImageCount + 1,
# #     imageFormat=surface_format.format,
# #     imageColorSpace=surface_format.colorSpace,
# #     imageExtent=vk.VkExtent2D(width=800, height=600),
# #     imageArrayLayers=1,
# #     imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
# #     imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
# #     preTransform=surface_caps.currentTransform,
# #     compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
# #     presentMode=present_mode,
# #     clipped=vk.VK_TRUE,
# #     oldSwapchain=vk.VK_NULL_HANDLE,
# # )
# # swapchain = vk.vkCreateSwapchainKHR(device, swapchain_info, None)
# # images = vk.vkGetSwapchainImagesKHR(device, swapchain)

# # # ---------- Render pass ----------
# # color_attachment = vk.VkAttachmentDescription(
# #     format=surface_format.format,
# #     samples=vk.VK_SAMPLE_COUNT_1_BIT,
# #     loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
# #     storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
# #     initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
# #     finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
# # )
# # color_ref = vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
# # subpass = vk.VkSubpassDescription(
# #     pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
# #     colorAttachmentCount=1,
# #     pColorAttachments=color_ref,
# # )
# # render_pass_info = vk.VkRenderPassCreateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
# #     attachmentCount=1,
# #     pAttachments=color_attachment,
# #     subpassCount=1,
# #     pSubpasses=subpass,
# # )
# # render_pass = vk.vkCreateRenderPass(device, render_pass_info, None)

# # # ---------- Framebuffers ----------
# # framebuffers = []
# # for img in images:
# #     view_info = vk.VkImageViewCreateInfo(
# #         sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
# #         image=img,
# #         viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
# #         format=surface_format.format,
# #         components=vk.VkComponentMapping(),
# #         subresourceRange=vk.VkImageSubresourceRange(
# #             aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
# #             baseMipLevel=0,
# #             levelCount=1,
# #             baseArrayLayer=0,
# #             layerCount=1,
# #         ),
# #     )
# #     img_view = vk.vkCreateImageView(device, view_info, None)
# #     fb_info = vk.VkFramebufferCreateInfo(
# #         sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
# #         renderPass=render_pass,
# #         attachmentCount=1,
# #         pAttachments=img_view,
# #         width=800,
# #         height=600,
# #         layers=1,
# #     )
# #     framebuffers.append(vk.vkCreateFramebuffer(device, fb_info, None))

# # # ---------- Command pool & buffer ----------
# # pool_info = vk.VkCommandPoolCreateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
# #     queueFamilyIndex=graphics_family,
# #     flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
# # )
# # cmd_pool = vk.vkCreateCommandPool(device, pool_info, None)
# # cmd_buf_alloc = vk.VkCommandBufferAllocateInfo(
# #     sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
# #     commandPool=cmd_pool,
# #     level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
# #     commandBufferCount=1,
# # )
# # cmd_buf = vk.vkAllocateCommandBuffers(device, cmd_buf_alloc)[0]

# # # ---------- Main loop ----------
# # while not glfw.window_should_close(window):
# #     glfw.poll_events()
# #     # Acquire next image
# #     img_index = vk.vkAcquireNextImageKHR(device, swapchain, vk.UINT64_MAX, vk.VK_NULL_HANDLE, vk.VK_NULL_HANDLE)[0]
# #     # Record command buffer
# #     begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
# #     vk.vkBeginCommandBuffer(cmd_buf, begin_info)
# #     clear_val = vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.1, 0.2, 0.3, 1.0]))
# #     render_pass_info = vk.VkRenderPassBeginInfo(
# #         sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
# #         renderPass=render_pass,
# #         framebuffer=framebuffers[img_index],
# #         renderArea=vk.VkRect2D(offset=vk.VkOffset2D(0, 0), extent=vk.VkExtent2D(800, 600)),
# #         clearValueCount=1,
# #         pClearValues=clear_val,
# #     )
# #     vk.vkCmdBeginRenderPass(cmd_buf, render_pass_info, vk.VK_SUBPASS_CONTENTS_INLINE)
# #     # No drawing commands – just clear
# #     vk.vkCmdEndRenderPass(cmd_buf)
# #     vk.vkEndCommandBuffer(cmd_buf)
# #     # Submit
# #     submit_info = vk.VkSubmitInfo(
# #         sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
# #         commandBufferCount=1,
# #         pCommandBuffers=cmd_buf,
# #     )
# #     vk.vkQueueSubmit(graphics_queue, 1, submit_info, vk.VK_NULL_HANDLE)
# #     vk.vkQueueWaitIdle(graphics_queue)
# #     # Present
# #     present_info = vk.VkPresentInfoKHR(
# #         sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
# #         swapchainCount=1,
# #         pSwapchains=swapchain,
# #         pImageIndices=img_index,
# #     )
# #     vk.vkQueuePresentKHR(graphics_queue, present_info)

# # # ---------- Cleanup ----------
# # vk.vkDeviceWaitIdle(device)
# # for fb in framebuffers:
# #     vk.vkDestroyFramebuffer(device, fb, None)
# # vk.vkDestroyRenderPass(device, render_pass, None)
# # vk.vkDestroySwapchainKHR(device, swapchain, None)
# # vk.vkDestroyDevice(device, None)
# # vk.vkDestroySurfaceKHR(instance, surface, None)
# # vk.vkDestroyInstance(instance, None)
# # glfw.destroy_window(window)
# # glfw.terminate()


# # import glfw, vulkan as vk, ctypes, sys

# # # ---------- GLFW window ----------
# # if not glfw.init(): sys.exit()
# # glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
# # win = glfw.create_window(800,600, "Vulkan on macOS", None, None)

# # # ---------- Vulkan setup ----------
# # app_info = vk.VkApplicationInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
# #  pApplicationName='PyVulkan',
# #  applicationVersion=vk.VK_MAKE_VERSION(1,0,0),
# #  pEngineName='NoEngine',
# #  engineVersion=vk.VK_MAKE_VERSION(1,0,0),
# #  apiVersion=vk.VK_API_VERSION_1_0)

# # exts = [vk.VK_KHR_SURFACE_EXTENSION_NAME,
# #  vk.VK_MVK_MACOS_SURFACE_EXTENSION_NAME]
# # inst_info = vk.VkInstanceCreateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
# #  pApplicationInfo=app_info,
# #  enabledExtensionCount=len(exts),
# #  ppEnabledExtensionNames=exts)
# # instance = vk.vkCreateInstance(inst_info, None)

# # # ---------- macOS surface ----------
# # surface_create_info = vk.VkMacOSSurfaceCreateInfoMVK(
# #  sType=vk.VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
# #  pView=ctypes.c_void_p(glfw.get_macos_window(win)))
# # surface = vk.vkCreateMacOSSurfaceMVK(instance, surface_create_info, None)

# # # ---------- Physical device & queue ----------
# # phys_devs = vk.vkEnumeratePhysicalDevices(instance)
# # phys = phys_devs[0]
# # queue_family_index = next(i for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(phys))
# #  if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)

# # # ---------- Logical device ----------
# # queue_info = vk.VkDeviceQueueCreateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
# #  queueFamilyIndex=queue_family_index,
# #  queueCount=1,
# #  pQueuePriorities=[1.0])
# # device_info = vk.VkDeviceCreateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
# #  queueCreateInfoCount=1,
# #  pQueueCreateInfos=queue_info)
# # device = vk.vkCreateDevice(phys, device_info, None)
# # queue = vk.vkGetDeviceQueue(device, queue_family_index,0)

# # # ---------- Command pool & buffer ----------
# # pool_info = vk.VkCommandPoolCreateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
# #  queueFamilyIndex=queue_family_index)
# # cmd_pool = vk.vkCreateCommandPool(device, pool_info, None)
# # alloc_info = vk.VkCommandBufferAllocateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
# #  commandPool=cmd_pool,
# #  level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
# #  commandBufferCount=1)
# # cmd_buf = vk.vkAllocateCommandBuffers(device, alloc_info)[0]

# # # ---------- Render pass (clear only) ----------
# # attachment = vk.VkAttachmentDescription(
# #  format=vk.VK_FORMAT_B8G8R8A8_UNORM,
# #  samples=vk.VK_SAMPLE_COUNT_1_BIT,
# #  loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
# #  storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
# #  initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
# #  finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
# # color_ref = vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
# # subpass = vk.VkSubpassDescription(colorAttachmentCount=1, pColorAttachments=color_ref)
# # render_pass_info = vk.VkRenderPassCreateInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
# #  attachmentCount=1, pAttachments=attachment,
# #  subpassCount=1, pSubpasses=subpass)
# # render_pass = vk.vkCreateRenderPass(device, render_pass_info, None)

# # # ---------- Main loop (clear to blue) ----------
# # while not glfw.window_should_close(win):
# #  glfw.poll_events()
# #  begin_info = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
# #  vk.vkBeginCommandBuffer(cmd_buf, begin_info)
# #  clear = vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.0,0.2,0.8,1.0]))
# #  rp_begin = vk.VkRenderPassBeginInfo(
# #  sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
# #  renderPass=render_pass,
# #  framebuffer=vk.VK_NULL_HANDLE, # placeholder – a real swapchain framebuffer is needed for a full app
# #  renderArea=vk.VkRect2D(offset=vk.VkOffset2D(0,0), extent=vk.VkExtent2D(800,600)),
# #  clearValueCount=1, pClearValues=clear)
# #  vk.vkCmdBeginRenderPass(cmd_buf, rp_begin, vk.VK_SUBPASS_CONTENTS_INLINE)
# #  vk.vkCmdEndRenderPass(cmd_buf)
# #  vk.vkEndCommandBuffer(cmd_buf)
# #  # Submit and present would go here (requires a swapchain). This minimal example just records commands.

# # vk.vkDeviceWaitIdle(device)
# # glfw.destroy_window(win)
# # glfw.terminate()
