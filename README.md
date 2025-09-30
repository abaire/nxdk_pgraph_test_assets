# nxdk_pgraph_test_assets
Raw asset files for abaire/nxdk_pgraph_tests

Assets are created using [Blender](https://www.blender.org/)

## Normal cubemap creation process

1. As a prerequisite, you may want to add a subdivision surface modifier with a high render count and zero viewport count. The render count is used when baking normals and the viewport count is used when exporting vertices. This allows the normals to carry detail not present in the low poly model.
1. Select the target object and go to Edit mode (`tab`)
   1. Select all faces (`A`) for the object for which normals should be created.
   1. Press `U` to open the unproject menu, and select "Smart UV Project"
1. Go to the "UV Editing" tab and confirm that the faces were properly unprojected.
1. Go to the "Shading" tab and add a new material if needed.
   1. Add a new "Image Texture" node (`shift + A`), leaving it disconnected from all other nodes.
   1. Click the "+ New" button and give the image a name (e.g., "MyModelNormalCubemap") and set the texture resolution (e.g. 256x256)
   1. In the right panel, go to the "Render" subtab and make sure the Render Engine is set to Cycles. You may also wish to change the device to "GPU compute" if it is set to "CPU".
   1. Expand the "Bake" panel in the list of subpanels.
   1. Set the "Bake Type" to "Normal" and the "Space" to "Object" so the normals are relative to the object origin.
   1. Make sure the texture node created previously is still selected and press the "Bake" button to generate the texture.
   1. When completed, save the texture to disk (e.g., through the hamburger menu next to the image name above the displayed bake results).

