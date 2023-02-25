


import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
from gpytoolbox import read_mesh, per_vertex_normals, per_face_normals  # just used to load a mesh and compute per-vertex normals


rng = np.random.default_rng()

rng = np.random.default_rng()
def normalizev(v):
    "normalize by element"
    norms = np.linalg.norm(v, axis=-1)
    norms = norms[:, np.newaxis]
    norms = np.where(norms==0,1,norms)
    norms = np.where(norms == np.inf, 1, norms)
    norms = np.where(norms == -np.inf, 1, norms)
    new_matrix = v / norms
    return new_matrix
def normalizevv(v):
    "normalize by element"
    row_sums = np.linalg.norm(v, axis=-1)
    row_sum = row_sums[:, np.newaxis]
    row_sum = np.where(row_sum!=0,v / row_sum,v)
    #new_matrix = v / row_sums[:, np.newaxis]
    return row_sum
def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; they must have the same
        size for gpytoolbox
        """
        if Os.shape[0] != Ds.shape[0]:
            if Ds.shape[0] == 1:
                self.Os = np.copy(Os)
                self.Ds = np.copy(Os)
                self.Ds[:, :] = Ds[:, :]
            if Os.shape[0] == 1:
                self.Ds = np.copy(Ds)
                self.Os = np.copy(Ds)
                self.Os[:, :] = Os[:, :]
        else:
            self.Os = np.copy(Os)
            self.Ds = np.copy(Ds)

    def __call__(self, t):
        """
        Computes an array of 3D locations given the distances
        to the points.
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return



def get_bary_coords(intersection, tri):
    denom = area(tri[:, 0], tri[:, 1], tri[:, 2])
    alpha_numerator = area(intersection, tri[:, 1], tri[:, 2])
    beta_numerator = area(intersection, tri[:, 0], tri[:, 2])
    alpha = alpha_numerator / denom
    beta = beta_numerator / denom
    gamma = 1 - alpha - beta
    barys = np.vstack((alpha, beta, gamma)).transpose()
    barys = np.where(np.isnan(barys), 0, barys)
    return barys


def area(t0, t1, t2):
    n = np.cross(t1 - t0, t2 - t0, axis=1)
    return np.linalg.norm(n, axis=1) / 2


def ray_mesh_intersect(origin, dir, tri):
    intersection = np.ones_like(dir) * -1
    intersection[:, 2] = np.Inf
    dir = dir[:, None]

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]  # (num_triangles, 3)
    s = origin[:, None] - tri[:, 0][None]
    s1 = np.cross(dir, e2)
    s2 = np.cross(s, e1)
    s1_dot_e1 = np.sum(s1 * e1, axis=2)
    results = np.ones((dir.shape[0], tri.shape[0])) * np.Inf

    if (s1_dot_e1 != 0).sum() > 0:
        coefficient = np.reciprocal(s1_dot_e1)
        alpha = coefficient * np.sum(s1 * s, axis=2)
        beta = coefficient * np.sum(s2 * dir, axis=2)
        cond_bool = np.logical_and(
            np.logical_and(
                np.logical_and(0 <= alpha, alpha < 1),
                np.logical_and(0 <= beta, beta < 1)
            ),
            np.logical_and(0 <= alpha + beta, alpha + beta < 1)
        )  # (num_rays, num_tri)
        e1_expanded = np.tile(e1[None], (dir.shape[0], 1, 1))  # (num_rays, num_tri, 3)
        dot_temp = np.sum(s1[cond_bool] * e1_expanded[cond_bool], axis=1)  # (num_rays,)
        results_cond1 = results[cond_bool]
        cond_bool2 = dot_temp != 0

        if cond_bool2.sum() > 0:
            coefficient2 = np.reciprocal(dot_temp)
            e2_expanded = np.tile(e2[None], (dir.shape[0], 1, 1))  # (num_rays, num_tri, 3)
            t = coefficient2 * np.sum(s2[cond_bool][cond_bool2] *
                                      e2_expanded[cond_bool][cond_bool2],
                                      axis=1)
            results_cond1[cond_bool2] = t
        results[cond_bool] = results_cond1
    results[results <= 0] = np.Inf
    hit_id = np.argmin(results, axis=1)
    min_val = np.min(results, axis=1)
    hit_id[min_val == np.Inf] = -1
    return min_val, hit_id





class Mesh(Geometry):
    def __init__(self, filename, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        self.Le = Le

        self.face_normals = per_face_normals(self.v, self.f, unit_norm=True)
        self.vertex_normals = per_vertex_normals(self.v, self.f)

        super().__init__()

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])

        hit_distances, triangle_hit_ids = ray_mesh_intersect(rays.Os, rays.Ds, self.v[self.f])
        intersections = rays.Os + hit_distances[:, None] * rays.Ds
        tris = self.v[self.f[triangle_hit_ids]]
        barys = get_bary_coords(intersections, tris)


        temp_normals = self.face_normals[triangle_hit_ids]
        temp_normals = self.vertex_normals[self.f[triangle_hit_ids]]
        temp_normals = temp_normals * barys[:,:,np.newaxis]
        temp_normals = np.sum(temp_normals,axis = 1)
        temp_normals = normalizev(temp_normals)


        temp_normals = np.where((triangle_hit_ids == -1)[:, np.newaxis],
                                hit_normals,
                                temp_normals)
        hit_normals = temp_normals

        return hit_distances, hit_normals


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays: output the
        intersection distances (set to np.inf if none), and unit hit
        normals (set to [np.inf, np.inf, np.inf] if none.)
        """
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:, :] = np.array([np.inf, np.inf, np.inf])



        L = rays.Os - self.c
        # get all terms
        A = np.sum(rays.Ds * rays.Ds, axis=-1)

        B = 2 * np.sum(L * rays.Ds, axis=-1)

        C = np.sum(L * L, axis=-1) - np.square(self.r)
        discriminant = np.square(B) - 4 * A * C
        discriminant = np.where(discriminant < 0, np.inf, discriminant)

        tplus = (-B + np.sqrt(discriminant)) / (2 * A)
        # get rid of negative values
        tplus = np.where(tplus < 0, np.inf, tplus)
        tminus = (-B - np.sqrt(discriminant)) / (2 * A)
        # get rid of negative values
        tminus = np.where(tminus < 0, np.inf, tminus)
        mint = np.minimum(tplus, tminus)

        discriminant = np.where(discriminant > 0, mint, discriminant)
        distances = np.where(discriminant == 0, -B / (2 * A), discriminant)
        # get normals
        temp = rays.Ds * distances[:, np.newaxis]
        hit_points = rays.Os + temp
        hit_normals = hit_points - self.c

        normals = normalizev(hit_normals)


        return distances, normals


# Enumerate the different importance sampling strategies we will implement
IMPLICIT_UNIFORM_SAMPLING, IMPLICIT_BRDF_SAMPLING = range(2)


class Scene(object):
    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_geometries(self, geometries):
        """
        Adds a list of geometries to the scene.

        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([geometries[i]])

        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter=False):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """


        origins = np.zeros((self.w * self.h, 3), dtype=np.float64)
        directions = np.zeros((self.w * self.h, 3), dtype=np.float64)
        vectorized_eye_rays = Rays(origins, directions)
        aspectratio = self.w / self.h

        scale = np.tan(np.deg2rad(self.fov) * 0.5)
        pixelx = np.linspace(0, self.w, num=self.w, endpoint=False, dtype=np.float64)
        pixely = np.linspace(0, self.h, num=self.h, endpoint=False, dtype=np.float64)
        pixelx, pixely = np.meshgrid(pixelx, pixely)
        if jitter:
            ndcx = (pixelx + rng.uniform(0.0, 1.0, size=pixelx.shape)) / self.w
            ndcy = (pixely + rng.uniform(0.0, 1.0, size=pixely.shape)) / self.h
        else:
            ndcx = (pixelx + 0.5) / self.w
            ndcy = (pixely + 0.5) / self.h
        pixelscreenx = 2 * ndcx - 1
        pixelscreeny = 1 - 2 * ndcy
        pixelcamerax = pixelscreenx * aspectratio * scale
        pixelcameray = pixelscreeny * scale
        # make the pixels
        pixelcameray = pixelcameray[:, :, np.newaxis]
        pixelcamerax = pixelcamerax[:, :, np.newaxis]

        pixelcam = np.concatenate((pixelcamerax, pixelcameray), axis=2)

        pixelcam = np.concatenate((pixelcam, np.ones((pixelcam.shape))), axis=2)

        pixelcam = pixelcam[:, :, :, np.newaxis]

        # variables for matrix
        zc = self.at - self.eye
        zc = normalize(self.at - self.eye)
        xc = normalize(np.cross(self.up, zc))
        yc = normalize(np.cross(zc, xc))
        eye = self.eye

        eye1 = eye[:, np.newaxis]
        zc = zc[:, np.newaxis]
        xc = xc[:, np.newaxis]
        yc = yc[:, np.newaxis]

        c2wmatrix = np.concatenate((xc, yc), axis=1)
        c2wmatrix = np.concatenate((c2wmatrix, zc), axis=1)
        c2wmatrix = np.concatenate((c2wmatrix, eye1), axis=1)
        c2wmatrix = np.concatenate((c2wmatrix, np.array([[0, 0, 0, 1]])), axis=0)

        pixelworld = np.matmul(c2wmatrix, pixelcam)

        pixelworld = np.delete(pixelworld, 3, 2)
        pixelworld = np.transpose(pixelworld, (0, 1, 3, 2))

        pixelworld = np.reshape(pixelworld, (pixelworld.shape[0] * pixelworld.shape[1], 3))
        pixelworld = pixelworld - eye


        new_matrix = normalizev(pixelworld)
        vectorized_eye_rays = Rays(self.eye[np.newaxis, :], new_matrix)


        return vectorized_eye_rays
        ### END CODE

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """


        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        hit_distances, hit_normals = self.geometries[-1].intersect(rays)
        hit_ids = hit_distances
        first_ind = len(self.geometries) - 1
        hit_ids = np.where(hit_distances < np.inf, first_ind, -1)
        # get points from distances

        inf = np.array([np.inf, np.inf, np.inf])
        newhitd = hit_distances[:, np.newaxis]
        inf = inf[np.newaxis, :]
        hit_normals = np.where(hit_distances[:, np.newaxis] < np.inf, hit_normals, inf)

        ###
        numsphere = len(self.geometries)
        a = hit_distances[0]
        for c in range(len(self.geometries) - 1):
            distances, normals = self.geometries[c].intersect(rays)
            # normals = rays.Os + rays.Ds * distances[:, np.newaxis] - self.geometries[c].c
            # normals = normalizev(normals)
            hit_normals = np.where(distances[:, np.newaxis] < hit_distances[:, np.newaxis], normals, hit_normals)
            hit_ids = np.where(distances < hit_distances, c, hit_ids)
            hit_distances = np.where(distances < hit_distances, distances, hit_distances)

        ###

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, num_bounces=3, sampling_type=IMPLICIT_BRDF_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                      np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                              np.array([0, 0, 0])[np.newaxis, :]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                  np.array([-1])))
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                      np.array([-1])))[ids]

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and(L_e != np.array([0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)



        # initialize arrays

        wi = np.zeros((self.h * self.w, 3))
        alpha = brdf_params[:,3][:,np.newaxis]
        psandpd = brdf_params[:,:3]

        # region initialize active
        #initialize active, active if hitpoint is nonlight,
        # false if hitpoint is light (L_e[:, 0][:,np.newaxis] != 0)/
        # out of scene(ids[:,np.newaxis]==-1)
        active = np.ones((self.h*self.w,1))
        active = np.where(np.logical_or(L_e[:, 0][:,np.newaxis] != 0,ids[:,np.newaxis]==-1), 0, active)
        # endregion

        #initialize throughput
        throughput = np.ones((self.h* self.w, 3),dtype = np.float64)
        #initialize dir and w
        #wo is only -ds for first bounce then wo = -wi
        wo = -eye_rays.Ds

        #into for loop
        for n in range(num_bounces):
            rand = np.random.rand(2, self.h * self.w)
            ndotwo = np.sum(normals * wo, axis=-1)
            wr = 2 * (ndotwo[:, np.newaxis]) * normals - wo

            # region M matrices
            # diffused M matrix
            dvz = normalizev(normals)
            dd = np.random.rand(self.h * self.w, 3)
            dd = normalizev(dd)
            dvx = normalizev(np.cross(dvz, dd))
            dvy = normalizev(np.cross(dvx, dvz))
            dM = np.transpose([np.transpose(dvx), np.transpose(dvy), np.transpose(dvz)])
            # glossy M
            gvz = normalizev(wr)  # (h*w,3)
            gd = np.random.rand(self.h * self.w, 3)
            gd = normalizev(gd)
            gvx = normalizev(np.cross(gvz, gd))
            gvy = normalizev(np.cross(gvx, gvz))
            gM = np.transpose([np.transpose(gvx), np.transpose(gvy), np.transpose(gvz)])
            # endregion

            # region wi generation
            # wiz
            wi[:, 2] = rand[0, :] ** (np.reciprocal(brdf_params[:, 3] + 1))
            R = np.sqrt(1 - wi[:, 2] ** 2)
            phi = 2 * np.pi * rand[1, :]
            # wix
            wi[:, 0] = R * np.cos(phi)
            # wiy
            wi[:, 1] = R * np.sin(phi)
            # wj
            dwi = (dM @ wi[:, :, np.newaxis]).reshape(wi.shape[0], wi.shape[1])
            gwi = (gM @ wi[:, :, np.newaxis]).reshape(wi.shape[0], wi.shape[1])
            wi = np.where(alpha == 1, dwi, gwi)
            # endregion

            # region max
            ndotw = np.sum(normals * wi, axis=-1)
            max = np.where(ndotw < 0, 0, ndotw)
            max = max[:, np.newaxis]
            # endregion

            #pfactor is 1 if alpha is 1, max if alpha>1
            pfactor = np.where(alpha ==1,1,max)
            #accumalate throughput
            throughput *= psandpd*(pfactor)

            # region recursion ray
            #construct recursion ray
            recur_ray = Rays(hit_points + shadow_ray_o_offset * normals, wi)
            distancesnew,normalsnew,idsnew = self.intersect(recur_ray)
            normalsnew = np.where(normalsnew != np.array([np.inf, np.inf, np.inf]),normalsnew, np.array([0, 0, 0]))
            hit_pointsnew = recur_ray(distancesnew)
            # endregion

            # region update variables wo, normals, ids, hitpoints
            wo = -recur_ray.Ds
            normals = normalsnew
            hit_points = hit_pointsnew
            ids = idsnew[:,np.newaxis]
            distances=distancesnew
            L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                                  np.array([0, 0, 0])[np.newaxis, :]))[idsnew]
            brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                          np.array([0, 0, 0, 1])[np.newaxis, :]))[idsnew]
            psandpd = brdf_params[:, :3]
            alpha = brdf_params[:, 3][:, np.newaxis]
            # endregion

            #accumalate throughput
            #throughput *= psandpd*(pfactor)

            # region update L
            #if x is at light -> active L +=Le(throughput) else L+=0,
            #hit -> (idsnew[:,np.newaxis] !=-1)
            #is light (L_e[:, 0][:,np.newaxis] != 0)
            b = np.logical_and(idsnew[:,np.newaxis] !=-1,L_e[:, 0][:,np.newaxis] != 0)
            a = np.logical_and(b ,active ==1)
            L1 = L.copy()
            L += L_e*throughput
            L = np.where(a,L, L1)
            # endregion

            # region update active T if T and
            # hitpoint hits(idsnew[:,np.newaxis]!=-1) and
            # not light (L_e[:, 0][:,np.newaxis] == 0)
            b =np.logical_and(idsnew[:,np.newaxis]!=-1,L_e[:, 0][:,np.newaxis] == 0)
            active = np.where(np.logical_and(active ==1,b),1,0)
            # endregion


        L = L.reshape((self.h, self.w, 3))
        return L
        ###

    def progressive_render_display(self, jitter=False, total_spp=20, num_bounces=3,
                                   sampling_type=IMPLICIT_BRDF_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)
        image_data = plt.imshow(L)

        for k in range(total_spp):
            o = k+1
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            plt.title(f"current spp: {k} of {total_spp}")
            L += self.render(vectorized_eye_rays,num_bounces=num_bounces, sampling_type=sampling_type)
            P = L/o
            image_data.set_data(np.clip(P,0,1))
            plt.pause(0.0001) # add a tiny delay between rendering passes


        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)





