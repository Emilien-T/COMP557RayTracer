# Emilien Taisne 261037777

import math
import glm
import numpy as np
import geometry as geom
import helperclasses as hc
from tqdm import tqdm

class Scene:
    def __init__(self,
                 width: int,
                 height: int,
                 jitter: bool,
                 samples: int,
                 eye_position: glm.vec3,
                 lookat: glm.vec3,
                 up: glm.vec3,
                 fov: float,
                 focal_length: float,
                 aperture: float,
                 dofSamples: int,
                 ambient: glm.vec3,
                 lights: list[hc.Light],
                 objects: list[geom.Geometry]
                 ):
        self.width = width  # width of image
        self.height = height  # height of image
        self.aspect = width / height  # aspect ratio
        self.jitter = jitter  # should rays be jittered
        self.samples = samples  # number of rays per pixel
        self.eye_position = eye_position  # camera position in 3D
        self.lookat = lookat  # camera look at vector
        self.up = up  # camera up position
        self.fov = fov  # camera field of view
        self.focal_length = focal_length  # focal length
        self.aperture = aperture # aperture
        self.dofSamples = dofSamples # depth of field samples
        self.ambient = ambient  # ambient lighting
        self.lights = lights  # all lights in the scene
        self.objects = objects  # all objects in the scene

    def BVH(self, objects):
        print("Building BVH")
        for object in objects:
            if isinstance(object, geom.Mesh):
                self.objects.append(geom.BVH(object.name, "bvh", object.materials, object, object.faces))
                self.objects.remove(object)
        print("BVH built")

    def render(self):

        image = np.zeros((self.height, self.width, 3)) # image with row,col indices and 3 channels, origin is top left

        cam_dir = self.eye_position - self.lookat
        distance_to_plane = 1.0
        top = distance_to_plane * math.tan(0.5 * math.pi * self.fov / 180)
        right = self.aspect * top
        bottom = -top
        left = -right

        w = glm.normalize(cam_dir)
        u = glm.cross(self.up, w)
        u = glm.normalize(u)
        v = glm.cross(w, u)

        for col in tqdm(range(self.width)):
            for row in range(self.height):
                # Per pixel loop
                colour = glm.vec3(0, 0, 0)
                if self.samples != 1:
                    for sx in range(self.samples):
                        for sy in range(self.samples):
                            if self.jitter:
                                x = col + (sx + np.random.rand()) / (self.samples)
                                y = row + (sy + np.random.rand()) / (self.samples)
                            else:
                                x = col + (sx + 0.5) / (self.samples)
                                y = row + (sy + 0.5) / (self.samples)

                            # TODO: Generate rays
                            viewing_ray = hc.Ray(self.eye_position, glm.normalize(w * (-distance_to_plane) +
                                                                                u * (left + (right - left) * (x + 0.5) / self.width) + #u * (right + (left - right) * (col + 0.5) / self.width)
                                                                                v * (top + (bottom - top) * (y + 0.5) / self.height)))

                            # TODO: Test for intersection with all objects
                            ClosestIntersection = self.Intersect(viewing_ray)
                            
                            # TODO: Perform shading computations on the intersection point
                            colour += self.Light(ClosestIntersection, viewing_ray)

                    colour = colour / (self.samples**2)

                else:
                    x = col
                    y = row

                    # TODO: Generate rays
                    viewing_ray = hc.Ray(self.eye_position, glm.normalize(w * (-distance_to_plane) +
                                                                        u * (left + (right - left) * (x + 0.5) / self.width) + #u * (right + (left - right) * (col + 0.5) / self.width)
                                                                        v * (top + (bottom - top) * (y + 0.5) / self.height)))

                    # TODO: Test for intersection with all objects
                    ClosestIntersection = self.Intersect(viewing_ray)

                    # TODO: Perform shading computations on the intersection point
                    colour += self.Light(ClosestIntersection, viewing_ray)

                image[row, col, 0] = max(0.0, min(1.0, colour.x))
                image[row, col, 1] = max(0.0, min(1.0, colour.y))
                image[row, col, 2] = max(0.0, min(1.0, colour.z))

        return image
    

    def Intersect(self, ray: hc.Ray):
        ClosestIntersection = hc.Intersection.default()
        for obj in self.objects:
            if obj == ray.obj:
                continue
            intersection = hc.Intersection.default()
            intersection = obj.intersect(ray, intersection)
            if intersection.t < ClosestIntersection.t:
                ClosestIntersection = intersection
        return ClosestIntersection

    def Light(self, closestIntersection: hc.Intersection, viewing_ray: hc.Ray, depth: int = 0, isDOF: bool = False):
        colour = glm.vec3(0, 0, 0)

        if depth > 3:
            return colour

        if closestIntersection.t < float("inf"):
            # Ambient lighting
            colour += self.ambient * closestIntersection.mat.diffuse
            for light in self.lights:
                colour += light.getContribution(closestIntersection, -viewing_ray.direction, self.objects)
        
            # Reflection
            if closestIntersection.mat.reflection != glm.vec3(0, 0, 0):
                reflection_ray = closestIntersection.reflect_ray(viewing_ray)
                reflection_intersection = self.Intersect(reflection_ray)
                reflection_colour = self.Light(reflection_intersection, reflection_ray, depth + 1, True)
                if reflection_intersection.t < float("inf"):
                    colour += closestIntersection.mat.reflection * reflection_colour
            
            # Refraction
            if closestIntersection.mat.transparency > 0:
                refraction_ray = closestIntersection.refract_ray(viewing_ray)
                if refraction_ray:
                    refraction_ray.obj = closestIntersection.obj
                    refraction_intersection = self.Intersect(refraction_ray)
                    refraction_colour = self.Light(refraction_intersection, refraction_ray, depth + 1, True)
                    colour += refraction_colour * closestIntersection.mat.transparency

            # Depth of field
            if self.aperture > 0 and not isDOF and closestIntersection.mat.transparency == 0:
                for i in range(self.dofSamples):
                    focal_point = viewing_ray.origin + self.focal_length * viewing_ray.direction
                    new_origin = viewing_ray.origin + glm.vec3(np.random.uniform(-self.aperture, self.aperture), np.random.uniform(-self.aperture, self.aperture), 0)
                    new_direction = glm.normalize(focal_point - new_origin)
                    new_ray = hc.Ray(new_origin, new_direction)
                    new_intersection = self.Intersect(new_ray)
                    new_colour = self.Light(new_intersection, new_ray, depth, True)
                    colour += new_colour
                
                colour = colour / (self.dofSamples + 1)


        return colour