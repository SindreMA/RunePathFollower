# RunePathFollower

This is a simple program that will follow a path in the game "RuneScape" and click on the minimap to move the character to the next point in the path.

Its using the [RuneLite](https://runelite.net/) client to find the game window and take a screenshot of the minimap.

And it is dependent on the plugin shortestpath to create the routes it will be following.

## Note

As of writing this, the program is just a concept demo and is not fully functional.


## How it works

The program finds the RuneLite client window, then takes a screenshot of the top right.

The screenshot is then processed in a few ways.
As the RuneLite sidebar can be open or closed, it will need to be removed from the image.

First it blacks out the RuneLite and windows borders from the image.
Then it fixes takes if from black to transparent, then it crops the image to not include any transparent pixels.

Now that it have a image of the map nicely sized, it will place a black circle in the center.
This will almost cover the the whole map.

Now it will do the opposite, only keeping a circle, and its slightly larger than the last.

This leaves us with a circle.

It will the scan the circle and look for a bunch of red pixels.

It has then the coordinates.

Then it will create a blue dot on the coordinates to indicate where it has found the path.

That is how far it goes.


## Next

Next step here would be to track the pixels moved, and reverse it to find out where we need to click.
Then of course click it.


## Problems

So there are some problems with this program.
Once you have moved into the line, line will show on two parts of the circle.
It will then go back and forward.

Sometimes it will creates routes with multiple paths too close to, and it will not know which one to follow.




