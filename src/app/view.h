#pragma once

// The arithmetic behind the preview's zoom, kept apart from the drawing code
// so it can be checked.
//
// It is three lines and it was wrong, in a way that is invisible when you
// nudge the wheel once and obvious when you turn it: the image walks away
// from the pointer instead of staying under it. Zoom that drifts is the kind
// of fault people work around without reporting, which is the best argument
// for it having a test rather than a careful reading.

namespace gsic {

// Where the image sits when it is centred in the pane at this scale, before
// any panning. One axis at a time; the two are independent.
inline float centred_base(float origin, float avail, float image_size, float scale) {
    return origin + (avail - image_size * scale) * 0.5f;
}

// The pan that keeps whatever is under `cursor` under `cursor` while the view
// scales from `old_scale` to `new_scale`.
//
// The error was using the new scale's centred base on both sides. The two
// bases differ by half the change in the image's size on screen -- around 90
// pixels per wheel click on a 2K image -- and the mistake compounds, so a few
// clicks put the picture somewhere else entirely.
inline float pan_for_zoom(float cursor, float pan, float origin, float avail,
                          float image_size, float old_scale, float new_scale) {
    const float base_old = centred_base(origin, avail, image_size, old_scale);
    const float base_new = centred_base(origin, avail, image_size, new_scale);
    // The image coordinate under the cursor, carried across the scale change.
    const float at = (cursor - pan - base_old) / old_scale;
    return cursor - at * new_scale - base_new;
}

// Where an image coordinate lands on screen, for checking the above.
inline float screen_position(float image_coord, float pan, float origin, float avail,
                             float image_size, float scale) {
    return centred_base(origin, avail, image_size, scale) + pan + image_coord * scale;
}

} // namespace gsic
