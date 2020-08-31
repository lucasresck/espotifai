# Overview

## What this is

**Espotifai** is our approach to the **playlist continuation problem**. That is, given a playlist, how to continuate it?

We know nowadays recommenders are extremely important, both for the greater production of content never seen before, and for the massive access provided by digital platforms. Music recommendation is no exception, so it is important to study and develop ways to match people and music they like or they will like.

We studied and implemented two algorithms that try to solve the problem of playlist continuation, inspired by [Kelen et al.](https://dl.acm.org/doi/10.1145/3267471.3267477) and [Pauws and Eggen](http://ismir2002.ircam.fr/proceedings/OKPROC02-FP07-4.pdf). Both of them use an idea of k-NN, but the former use a similarity among playlists and the latter use a similarity among tracks.

## Motivation

Recommendation is pretty intrigating, because we have to know by advance what
the person would like to listen in a specific time. That's a hard task, due to
the human complexity, however we can simplify this problem and we like that!
Math is all about that!! And of course, we love music! 

## Our goals

Given an incomplete playlist, we should complete it. Also, we should deal with the problem of the leak of playlist data on the internet, so as with the computational problems that appear.

This is our final project for [Foundations of Data Science](https://emap.fgv.br/disciplina/mestrado/fundamentos-de-ciencia-de-dados), a Mathematical Modelling Master's subject at Getulio Vargas Foundation (FGV).

Group: [Lucas Emanuel Resck Domingues](https://github.com/lucasresck) and [Lucas Machado Moschen](https://github.com/lucasmoschen).
Professor: [Dr. Jorge Poco](https://github.com/jpocom).
