import helperclasses as hc
import glm
import igl
from perlin_noise import PerlinNoise

noise = PerlinNoise(octaves=20)

class Geometry:
    def __init__(self, name: str, gtype: str, materials: list[hc.Material]):
        self.name = name
        self.gtype = gtype
        self.materials = materials

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect.obj = self
        return intersect
    
    def bumpMap(self, intersect: hc.Intersection):
        if intersect.mat.hasBumpMap:
            noisemap = noise([intersect.position.x, intersect.position.y, intersect.position.z])
            perturbed_normal = glm.normalize(intersect.normal + glm.vec3(noisemap, noisemap, noisemap)*0.5)
            intersect.normal = perturbed_normal
        return intersect

class Sphere(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], center: glm.vec3, radius: float):
        super().__init__(name, gtype, materials)
        self.center = center
        self.radius = radius

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        ray_to_center = self.center - ray.origin
        a = glm.dot(ray.direction, ray.direction)
        b = -2 * glm.dot(ray.direction, ray_to_center)
        c = glm.dot(ray_to_center, ray_to_center) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return intersect
        t1 = (-b + glm.sqrt(discriminant)) / (2 * a)
        t2 = (-b - glm.sqrt(discriminant)) / (2 * a)
        if min(t1, t2) < 0:
            return intersect
        intersect.t = min(t1, t2)
        intersect.position = ray.getPoint(intersect.t)
        intersect.normal = glm.normalize(intersect.position - self.center)
        intersect.mat = self.materials[0]
        intersect = self.bumpMap(intersect)
        return intersect


class Plane(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], point: glm.vec3, normal: glm.vec3):
        super().__init__(name, gtype, materials)
        self.point = point
        self.normal = normal

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        DdotN = glm.dot(ray.direction, self.normal)
        if DdotN == 0:
            return intersect
        t = glm.dot(self.point - ray.origin, self.normal) / DdotN
        if t < 0:
            return intersect
        intersect.t = t
        intersect.position = ray.getPoint(intersect.t)
        intersect.normal = self.normal
        if len(self.materials) == 1:
            intersect.mat = self.materials[0]
        else:
            # Create a checkerboard pattern
            x = round(intersect.position.x + 0.5) % 2
            z = round(intersect.position.z + 0.5) % 2
            if x == 0:
                if z == 0:
                    intersect.mat = self.materials[0]
                else:
                    intersect.mat = self.materials[1]
            else:
                if z == 0:
                    intersect.mat = self.materials[1]
                else:
                    intersect.mat = self.materials[0]
        intersect = self.bumpMap(intersect)
        return intersect

class AABB(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], minpos: glm.vec3, maxpos: glm.vec3):
        # dimension holds information for length of each size of the box
        super().__init__(name, gtype, materials)
        self.minpos = minpos
        self.maxpos = maxpos

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        # Slab method
        tmin = glm.vec3(0,0,0)
        tmax = glm.vec3(0,0,0)
        # getting all the t values for each axis
        for i in range(3):
            if ray.direction[i] == 0:
                if ray.origin[i] < self.minpos[i] or ray.origin[i] > self.maxpos[i]:
                    return intersect
                else:
                    tmin[i] = -float('inf')
                    tmax[i] = float('inf')
            else:
                tmin[i] = (self.minpos[i] - ray.origin[i]) / ray.direction[i]
                tmax[i] = (self.maxpos[i] - ray.origin[i]) / ray.direction[i]
        
        # getting the t value for the intersection
        tlow = glm.vec3(min(tmin[0], tmax[0]), min(tmin[1], tmax[1]), min(tmin[2], tmax[2]))
        thigh = glm.vec3(max(tmin[0], tmax[0]), max(tmin[1], tmax[1]), max(tmin[2], tmax[2]))

        tminf = max(tlow[0], tlow[1], tlow[2])
        tmaxf = min(thigh[0], thigh[1], thigh[2])

        if tminf > tmaxf or tmaxf < 0:
            return intersect

        intersect.t = tminf
        intersect.position = ray.getPoint(intersect.t)
        if self.materials == []:
            intersect.mat = None
        intersect.mat = self.materials[0]

        # getting the normal of the surface
        if intersect.t in tmin:
            if intersect.t == tmin[0]:
                intersect.normal = glm.vec3(-1,0,0)
            elif intersect.t == tmin[1]:
                intersect.normal = glm.vec3(0,-1,0)
            else:
                intersect.normal = glm.vec3(0,0,-1)
        else:
            if intersect.t == tmax[0]:
                intersect.normal = glm.vec3(1,0,0)
            elif intersect.t == tmax[1]:
                intersect.normal = glm.vec3(0,1,0)
            else:
                intersect.normal = glm.vec3(0,0,1)

        intersect = self.bumpMap(intersect)
        
        return intersect

class Mesh(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], translate: glm.vec3, scale: float,
                 filepath: str):
        super().__init__(name, gtype, materials)
        verts, _, norms, self.faces, _, _ = igl.read_obj(filepath)
        self.verts = []
        self.norms = []
        for v in verts:
            self.verts.append((glm.vec3(v[0], v[1], v[2]) + translate) * scale)
        for n in norms:
            self.norms.append(glm.vec3(n[0], n[1], n[2]))

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        # Test for intersection with all faces
        for face in self.faces:
            a = self.verts[face[0]]
            b = self.verts[face[1]]
            c = self.verts[face[2]]

            # Calculate the normal of the face
            normal = glm.normalize(glm.cross(b - a, c - a))

            # Calculate the intersection point on the plane
            DdotN = glm.dot(ray.direction, normal)
            if DdotN == 0:
                continue
            t = glm.dot(a - ray.origin, normal) / DdotN
            if t < 0:
                continue
            
            # Check if the intersection point is inside the triangle
            P = ray.getPoint(t)
            if  (glm.dot(glm.cross(b - a, P - a), normal) >= 0 and
                 glm.dot(glm.cross(c - b, P - b), normal) >= 0 and
                 glm.dot(glm.cross(a - c, P - c), normal) >= 0):
                if intersect.t > t:
                    intersect.t = t
                    intersect.position = P
                    intersect.normal = normal
                    intersect.mat = self.materials[0]
                    intersect = self.bumpMap(intersect)
        return intersect

class Node(Geometry):
    def __init__(self, name: str, gtype: str, M: glm.mat4, materials: list[hc.Material]):
        super().__init__(name, gtype, materials)        
        self.children: list[Geometry] = []
        self.M = M
        self.Minv = glm.inverse(M)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        # Transform the ray into the object space
        newRay = hc.Ray(
            glm.vec3(self.Minv * glm.vec4(ray.origin, 1)),
            glm.vec3(self.Minv * glm.vec4(ray.direction, 0)))
        # Test for intersection with all children
        for child in self.children:
            newIntersect = hc.Intersection.default()
            newIntersect = child.intersect(newRay, newIntersect)
            if intersect.t > newIntersect.t:
                intersect = newIntersect
        
        if intersect.t < float("inf"):
            # Transform the intersection back to the world space
            intersect.position = glm.vec3(self.M * glm.vec4(intersect.position, 1))
            intersect.normal = glm.vec3(glm.normalize(self.M * glm.vec4(intersect.normal, 0)))
        return intersect
    
class BVH(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], mesh: Mesh, faces: list[glm.vec3]):
        super().__init__(name, gtype, materials)
        self.mesh = mesh
        self.faces = list(faces)
        self.left = None
        self.right = None
        self.minpos = glm.vec3(float('inf'), float('inf'), float('inf'))
        self.maxpos = glm.vec3(-float('inf'), -float('inf'), -float('inf'))
        self.build()
    
    def build(self):
        # Calculate the bounding box
        for face in self.faces:
            a = self.mesh.verts[face[0]]
            b = self.mesh.verts[face[1]]
            c = self.mesh.verts[face[2]]
            self.minpos = glm.min(self.minpos, glm.min(a, glm.min(b, c)))
            self.maxpos = glm.max(self.maxpos, glm.max(a, glm.max(b, c)))
        # Split
        if len(self.faces) > 1:
            # Calculate the longest axis
            diff = self.maxpos - self.minpos
            axis = 0
            if diff.y > diff.x:
                axis = 1
            if diff.z > diff.y and diff.z > diff.x:
                axis = 2
            # Sort the faces along the axis
            self.faces.sort(key=(lambda f: (self.mesh.verts[f[0]][axis] + self.mesh.verts[f[1]][axis] + self.mesh.verts[f[2]][axis]) / 3))
            # Split the faces
            self.left = BVH(self.name + "_left", "bvh", self.materials, self.mesh, self.faces[:len(self.faces)//2])
            self.left.build()
            self.right = BVH(self.name + "_right", "bvh", self.materials, self.mesh, self.faces[len(self.faces)//2:])
            self.right.build()
        else:
            self.left = self.right = None


    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        if(len(self.faces) == 1):
            face = self.faces[0]
            a = self.mesh.verts[face[0]]
            b = self.mesh.verts[face[1]]
            c = self.mesh.verts[face[2]]

            # Calculate the normal of the face
            normal = glm.normalize(glm.cross(b - a, c - a))

            # Calculate the intersection point on the plane
            DdotN = glm.dot(ray.direction, normal)
            if DdotN == 0:
                return intersect
            t = glm.dot(a - ray.origin, normal) / DdotN
            if t < 0:
                return intersect
            
            # Check if the intersection point is inside the triangle
            P = ray.getPoint(t)
            if  (glm.dot(glm.cross(b - a, P - a), normal) >= 0 and
                    glm.dot(glm.cross(c - b, P - b), normal) >= 0 and
                    glm.dot(glm.cross(a - c, P - c), normal) >= 0):
                if intersect.t > t:
                    intersect.t = t
                    intersect.position = P
                    intersect.normal = normal
                    intersect.mat = self.materials[0]
                    intersect = self.bumpMap(intersect)
            return intersect
        
        # Slab method
        tmin = glm.vec3(0,0,0)
        tmax = glm.vec3(0,0,0)
        # getting all the t values for each axis
        for i in range(3):
            if ray.direction[i] == 0:
                if ray.origin[i] < self.minpos[i] or ray.origin[i] > self.maxpos[i]:
                    return intersect
                else:
                    tmin[i] = -float('inf')
                    tmax[i] = float('inf')
            else:
                tmin[i] = (self.minpos[i] - ray.origin[i]) / ray.direction[i]
                tmax[i] = (self.maxpos[i] - ray.origin[i]) / ray.direction[i]
        
        # getting the t value for the intersection
        tlow = glm.vec3(min(tmin[0], tmax[0]), min(tmin[1], tmax[1]), min(tmin[2], tmax[2]))
        thigh = glm.vec3(max(tmin[0], tmax[0]), max(tmin[1], tmax[1]), max(tmin[2], tmax[2]))

        tminf = max(tlow[0], tlow[1], tlow[2])
        tmaxf = min(thigh[0], thigh[1], thigh[2])

        if tminf > tmaxf or tmaxf < 0:
            return intersect

        # Test for intersection with the children
        if self.left is not None:
            intersect = self.left.intersect(ray, intersect)
        if self.right is not None:
            intersect = self.right.intersect(ray, intersect)
        return intersect
    
class Quadradic(Geometry):
    def __init__(self, name: str, gtype: str, materials: list[hc.Material], M: glm.mat4):
        super().__init__(name, gtype, materials)
        self.M = M
        self.Minv = glm.inverse(M)

    def intersect(self, ray: hc.Ray, intersect: hc.Intersection):
        intersect = super().intersect(ray, intersect)
        direction = glm.vec4(ray.direction, 0)
        origin = glm.vec4(ray.origin, 1)
        a = glm.dot(direction, self.M * direction)
        b = glm.dot(direction, self.M * origin) + glm.dot(origin, self.M * direction)
        c = glm.dot(origin, self.M * origin)
        if a < 1e-6 and a > -1e-6:
            t = -c / b
        else:
            descriminant = b * b - 4 * a * c
            if descriminant < 0:
                return intersect
            t1 = (-b + glm.sqrt(descriminant)) / (2*a)
            t2 = (-b - glm.sqrt(descriminant)) / (2*a)
            t = min(t1, t2)
        if t < 0:
            return intersect
        intersect.t = t
        intersect.position = ray.getPoint(intersect.t)
        intersect.normal = glm.normalize(glm.vec3(intersect.position.x * 2 * self.M[0][0] + intersect.position.y * (self.M[0][1] + self.M[1][0]) + intersect.position.z * (self.M[0][2] + self.M[2][0]) + self.M[0][3] + self.M[3][0],
                                     intersect.position.x * (self.M[0][1] + self.M[1][0]) + intersect.position.y * 2 * self.M[1][1] + intersect.position.z * (self.M[1][2] + self.M[2][1]) + self.M[1][3] + self.M[3][1],
                                     intersect.position.x * (self.M[0][2] + self.M[2][0]) + intersect.position.y * (self.M[1][2] + self.M[2][1]) + intersect.position.z * 2 * self.M[2][2] + self.M[2][3] + self.M[3][2]))
        intersect.mat = self.materials[0]
        intersect = self.bumpMap(intersect)
        return intersect