// Release numbering for .github/workflows/release.yml.
//
// Called through actions/github-script, which supplies `github` (an
// authenticated Octokit), `context` and `core`. Kept in a file of its own so
// the logic can be exercised with plain `node` and a fake `github`.

const VERSION_TAG = /^v(\d+)\.(\d+)\.(\d+)$/;

/** `v1.2.3` -> [1, 2, 3]; anything else (`v1.2.3-rc1`, `nightly`) -> null. */
function parse(tag) {
  const match = VERSION_TAG.exec(tag);
  return match ? match.slice(1).map(Number) : null;
}

function compare(a, b) {
  for (let i = 0; i < 3; i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

function format(version) {
  return `v${version.join('.')}`;
}

/**
 * Decide which tag this run releases and set it as the `tag` output.
 *
 * - A pushed `v*` tag is released as it is.
 * - A push to a branch gets the next patch version after the highest
 *   `vX.Y.Z` tag in the repository; the tag is created here, on the pushed
 *   commit. Creating a ref is atomic on GitHub's side, so two runs racing for
 *   the same number cannot both get it: the loser moves on to the next one.
 * - A commit that already carries a version tag is not numbered twice. On a
 *   re-run that tag is this run's own from the first attempt and is reused; on
 *   a first attempt someone pushed it by hand, its own run publishes it, and
 *   the output is empty (the workflow then skips the publish job).
 */
async function pickTag({ github, context, core }) {
  if (context.ref.startsWith('refs/tags/')) {
    const tag = context.ref.slice('refs/tags/'.length);
    core.setOutput('tag', tag);
    return tag;
  }

  const { owner, repo } = context.repo;
  const tags = await github.paginate(github.rest.repos.listTags, { owner, repo, per_page: 100 });
  const versions = [];
  const here = [];
  for (const tag of tags) {
    const version = parse(tag.name);
    if (!version) continue;
    versions.push(version);
    if (tag.commit.sha === context.sha) here.push(version);
  }
  versions.sort(compare);
  here.sort(compare);

  if (here.length > 0) {
    const tag = format(here[here.length - 1]);
    const attempt = Number(process.env.GITHUB_RUN_ATTEMPT || '1');
    if (attempt > 1) {
      core.notice(`re-run: ${context.sha.slice(0, 7)} is already ${tag}, publishing to it again`);
      core.setOutput('tag', tag);
      return tag;
    }
    core.notice(`${context.sha.slice(0, 7)} already carries ${tag}; that tag's own run publishes it`);
    core.setOutput('tag', '');
    return '';
  }

  // No version tag yet: the first release is v0.1.0.
  const next = versions.length > 0 ? [...versions[versions.length - 1]] : [0, 1, -1];
  let refused = null;
  for (let tries = 0; tries < 50; tries += 1) {
    next[2] += 1;
    const tag = format(next);
    try {
      await github.rest.git.createRef({ owner, repo, ref: `refs/tags/${tag}`, sha: context.sha });
    } catch (error) {
      // 422 "Reference already exists": another push's run took this number.
      if (error.status !== 422) throw error;
      refused = error;
      continue;
    }
    core.notice(`${context.sha.slice(0, 7)} is ${tag}`);
    core.setOutput('tag', tag);
    return tag;
  }
  throw new Error(`no free version tag after ${format(next)}: ${refused && refused.message}`);
}

/**
 * Set the `make_latest` output for the release of `tag`.
 *
 * Runs for different pushes build in parallel and may publish out of order, so
 * "the release published last" is not necessarily the newest one. `false` when
 * a higher version is already published, `true` otherwise; a tag that is not a
 * plain `vX.Y.Z` leaves the choice to GitHub (`legacy`).
 */
async function makeLatest({ github, context, core }, tag) {
  const mine = parse(tag);
  if (!mine) {
    core.setOutput('make_latest', 'legacy');
    return 'legacy';
  }
  const { owner, repo } = context.repo;
  // Newest first; a higher version than this run's is among the most recent.
  const { data: releases } = await github.rest.repos.listReleases({ owner, repo, per_page: 100 });
  const newer = releases.some((release) => {
    if (release.draft || release.prerelease) return false;
    const version = parse(release.tag_name);
    return version !== null && compare(version, mine) > 0;
  });
  const value = newer ? 'false' : 'true';
  core.setOutput('make_latest', value);
  return value;
}

module.exports = { pickTag, makeLatest, parse, compare, format };
