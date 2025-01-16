# Emilien Taisne 261037777

import glm

class Ray:
    def __init__(self, o: glm.vec3, d: glm.vec3, obj = None):
        self.origin = o
        self.direction = d
        self.obj = obj

    def getDistance(self, point: glm.vec3):
        return glm.length(point - self.origin)

    def getPoint(self, t: float):
        return self.origin + self.direction * t

class Material:
    def __init__(self, name: str, diffuse: glm.vec3, specular: glm.vec3, shininess: float, reflection: glm.vec3, ior = 1.0, transparency = 0.0, hasBumpMap = False):
        self.name = name
        self.diffuse = diffuse      # kd diffuse coefficient
        self.specular = specular    # ks specular coefficient
        self.shininess = shininess  # specular exponent 
        self.reflection = reflection # reflection coefficient
        self.ior = ior              # index of refraction
        self.transparency = transparency
        self.hasBumpMap = hasBumpMap

class Intersection:
    def __init__(self, t: float, normal: glm.vec3, position: glm.vec3, material: Material, obj = None):
        self.t = t
        self.normal = normal
        self.position = position
        self.mat = material
        self.obj = obj

    @staticmethod
    def default(): # create an empty intersection record with t = inf
        t = float("inf")
        normal = None 
        position = None 
        mat = None 
        return Intersection(t, normal, position, mat)
    
    def reflect_ray(self, incoming_ray: Ray):
        reflection_direction = glm.reflect(incoming_ray.direction, self.normal)
        return Ray(self.position + 0.001 * reflection_direction, reflection_direction)
    
    def refract_ray(self, incoming_ray: Ray):
        cosi = max(-1, min(1, glm.dot(incoming_ray.direction, self.normal)))
        etai = 1
        etat = self.mat.ior
        n = self.normal
        if cosi < 0:
            cosi = -cosi
        else:
            etai, etat = etat, etai
            n = -self.normal
        eta = etai / etat
        k = 1 - eta * eta * (1 - cosi * cosi)
        if k < 0:
            return None  # Total internal reflection
        else:
            refraction_direction = eta * incoming_ray.direction + (eta * cosi - glm.sqrt(k)) * n
            return Ray(self.position - 0.001 * refraction_direction, refraction_direction)

class Light:
    def __init__(self, ltype: str, name: str, colour: glm.vec3, vector: glm.vec3, attenuation: glm.vec3):
        self.name = name
        self.type = ltype       # type is either "point" or "directional"
        self.colour = colour    # colour and intensity of the light
        self.vector = vector    # position, or normalized direction towards light, depending on the light type
        self.attenuation = attenuation # attenuation coeffs [quadratic, linear, constant] for point lights
    
    def getContribution(self, intersection: Intersection, eye, objects):
        # Compute the contribution of this light to the shading at the intersection point
        if self.type == "point":
            light_distance = glm.length(self.vector - intersection.position)
            attenuation = 1 / (self.attenuation[0] * light_distance * light_distance + # quadratic
                               self.attenuation[1] * light_distance +                  # linear
                               self.attenuation[2])                                    # constant
            color = self.colour * attenuation
            light_direction = glm.normalize(self.vector - intersection.position)
        else:
            color = self.colour
            light_direction = -self.vector

        # Check for shadow
        start_position = intersection.position + 0.001 * light_direction
        shadow_ray = Ray(start_position, light_direction)
        shadow_intersection = Intersection.default()
        for obj in objects:
            shadow_intersection = obj.intersect(shadow_ray, shadow_intersection)
            if shadow_intersection.t < float("inf"):
                # print(intersection.mat.name, obj.name, shadow_intersection.t)
                return glm.vec3(0, 0, 0)

        # Diffuse Lambertian (kd * I * max(0, N.L))
        Ld = intersection.mat.diffuse * color * max(0, glm.dot(intersection.normal, light_direction))
        # Blinn-Phong Specular (ks * I * max(0, N.H)^shininess)
        Ls = intersection.mat.specular * color * max(0, glm.dot(intersection.normal, glm.normalize(light_direction + eye))) ** intersection.mat.shininess
        return Ld + Ls

