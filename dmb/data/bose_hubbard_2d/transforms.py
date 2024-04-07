class GaussianNoise:

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={}, std={})".format(
            self.mean, self.std)


# Transorm for symmetry of the square
class SquareSymmetryGroupAugmentations(object):

    def __call__(
        self, xy: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(xy, tuple):
            x = xy[0]
            y = xy[1]

        else:
            x = xy
            y = None

        def map_if_not_none(
                fn: Callable[[torch.Tensor], torch.Tensor],
                x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            else:
                return fn(x)

        # with p=1/8 each choose one symmetry transform at random and apply it
        rnd = np.random.rand()

        if rnd < 1 / 8:  # unity
            pass
        elif rnd < 2 / 8:  # rotate 90 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 1, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 3 / 8:  # rotate 180 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 2, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 4 / 8:  # rotate 270 left
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.rot90(x, 3, [-2, -1]), xy),
                (x, y),
            )
        elif rnd < 5 / 8:  # flip x
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-2]), xy),
                (x, y))
        elif rnd < 6 / 8:  # flip y
            x, y = map(
                lambda xy: map_if_not_none(lambda x: torch.flip(x, [-1]), xy),
                (x, y))
        elif rnd < 7 / 8:  # reflection x=y
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.transpose(x, -2, -1), xy),
                (x, y),
            )
        else:  # reflection x=-y
            x, y = map(
                lambda xy: map_if_not_none(
                    lambda x: torch.flip(torch.transpose(x, -2, -1), [-2, -1]),
                    xy),
                (x, y),
            )

        if y is None:
            return x
        else:
            return x, y

    def __repr__(self):
        return self.__class__.__name__ + "()"


class TupleWrapperInTransform(object):

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform

    def __call__(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(x, tuple):
            return self.transform(x[0]), x[1]
        else:
            return self.transform(x)

    def __repr__(self):
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__(
        )


class TupleWrapperOutTransform(object):

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.transform = transform

    def __call__(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(x, tuple):
            return x[0], self.transform(x[1])
        else:
            return self.transform(x)

    def __repr__(self):
        return self.__class__.__name__ + "()" + "\n" + self.transform.__repr__(
        )
